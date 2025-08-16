from functions import *
from model import *
from utils import *

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib as plt
import math
from tqdm import tqdm
import os
import numpy as np
from collections import Counter
import regex as re
import torch.backends
import copy


if __name__ == "__main__":

    PAD_TOKEN = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_data = load_data(os.path.join('dataset','ATIS','train.json'))
    test_data = load_data(os.path.join('dataset','ATIS','test.json'))
    train_raw, dev_raw, test_raw, y_train, y_dev, y_test = get_splits(train_data, test_data)

    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, cutoff=0)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Parameters setting ==========================================================================
    hid_size = 400
    emb_size = 500
    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient
    n_epochs = 200
    runs = 5
    use_dropout = False
    use_bidirectional = False
    patience_init = 3
    # Parameters setting ==========================================================================

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    slot_f1s = []
    intent_acc = []

    # Run the experiment for a given number of runs
    for x in tqdm(range(0, runs)):
        
        # Initialize the model for Intent and Slot classification
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN, use_drop=use_dropout, use_bidirectional=use_bidirectional).to(device)
        model.apply(init_weights)

        # Set optimizer and loss functions
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        # Early stopping parameters
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0.0

        # Training loop
        for x in range(1,n_epochs+1):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model, clip=clip)
            
            # Evaluate every 5 epochs
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                # Save best model based on F1 score
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = copy.deepcopy(model).to(device)
                    patience = patience_init
                else:
                    patience -= 1
                
                if patience <= 0: # Early stopping with patient
                    break
        
        # Load best model for testing
        best_model.to(device)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
        
        model_name = build_model_name(
            lr=lr,
            slot_f1s=slot_f1s,
            intent_acc=intent_acc,
            bidirectional=use_bidirectional,
            dropout=use_dropout
        )
        
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)

    save_experiment_results(model, optimizer=optimizer, n_epochs=n_epochs, lr=lr, hid_size=hid_size, emb_size=emb_size, slot_f1s=slot_f1s, intent_acc=intent_acc,
                            losses_train=losses_train, losses_dev=losses_dev, runs=runs, dropout=use_dropout, bidirectional=use_bidirectional, patience=patience_init, model_name=model_name)