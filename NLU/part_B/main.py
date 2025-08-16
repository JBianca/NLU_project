from functions import *
from utils import *
from model import *
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
import torch
from tqdm import tqdm
import os
import copy
from transformers import BertTokenizer
from model import JointBert


if __name__ == "__main__":

    PAD_TOKEN = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load dataset
    train_data = load_data(os.path.join('dataset','ATIS','train.json'))
    test_data = load_data(os.path.join('dataset','ATIS','test.json'))
    train_raw, dev_raw, test_raw, y_train, y_dev, y_test = get_splits(train_data, test_data)

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, tokenizer, lang)
    dev_dataset = IntentsAndSlots(dev_raw, tokenizer, lang)
    test_dataset = IntentsAndSlots(test_raw,tokenizer, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Parameters setting ==========================================================================
    hid_size = 768
    emb_size = 300
    lr = 0.0002
    clip = 5 # Clip the gradient
    n_epochs = 50
    dropout = 0.1
    patience = 5
    # Parameters setting ==========================================================================

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    slot_f1s = []
    intent_acc = []

    # Initialize the model for Intent and Slot classification
    model = JointBert(hid_size, out_slot, out_int, dropout=dropout).to(device)
    model.apply(init_weights)

    # Set optimizer and loss functions
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0.0
    
    # Training loop
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model, clip=clip)
        if x % 1 == 0:
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang, tokenizer)
            losses_dev.append(np.asarray(loss_dev).mean())
            f1 = results_dev['total']['f']
            
            # Save best model based on F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to(device)
                patience = patience
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patient
                break 

    # Load best model for testing
    best_model.to(device)
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang, tokenizer)
    intent_acc.append(intent_test['accuracy'])
    slot_f1s.append(results_test['total']['f'])

    model_name = build_model_name(
        lr=lr,
        slot_f1s=slot_f1s,
        intent_acc=intent_acc,
        dropout=dropout
    )
        
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)

    save_experiment_results(model, optimizer=optimizer, n_epochs=len(sampled_epochs), lr=lr,  slot_f1s=slot_f1s, intent_acc=intent_acc,
                            losses_train=losses_train, losses_dev=losses_dev, dropout=dropout, patience=patience, model_name=model_name, best_f1=best_f1)