from functions import *
from model import *
from utils import *

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
import math
from tqdm import tqdm
import copy
import argparse

if __name__ == "__main__":
    
    ######################################################################################################### data preparation 
    # Use cuda if available, otherwise cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    # Vocab is computed only on training set 
    # We add two special tokens end of sentence and padding 
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    '''
    --> try with all the combinations
    [(32, 64), (32, 128), (64, 64), (64, 128), (128, 64), (128, 128)]
    '''
    batch_train = 64 
    batch_dev_test = 128

    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=batch_train, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_dev_test, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=batch_dev_test, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    ######################################################################################################### parameters configuration

    '''
    Experiment also with a smaller or bigger model by changing hid and emb sizes
    A large model tends to overfit
    Don't forget to experiment with a lower training batch size
    Increasing the back propagation steps can be seen as a regularization step
    With SGD try with an higher learning rate (> 1 for instance)
    ''' 

    hid_size = 200
    emb_size = 300
    vocab_len = len(lang.word2id)
    lr = 0.001 # [0.05, 0.01] for SGD; [0.001] for Adam
    clip = 5 # Clip the gradient
    n_epochs = 100
    patience = 3
    out_dropout = 0.0
    emb_dropout = 0.0
    weight_decay = 0.0
    n_layers = 1

    ##############################################################
   
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    perplexities = [] # add pp vector for plot
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs+1))
    
    ######################################################################################################### model
    '''
    modalità: uno dei tre in base al parametro
        model
        optimizer
    '''
    parser = argparse.ArgumentParser(description="Train a language model with configurable model and optimizer.")

    parser.add_argument("--model", type=str, required=True,
                        choices=["rnn", "lstm", "lstm_dropout"],
                        help="Choose the model architecture.")
    parser.add_argument("--optimizer", type=str, required=True,
                        choices=["sgd", "adamw"],
                        help="Choose the optimizer.")

    args = parser.parse_args()

    model_map = {
        "rnn": LM_RNN,
        "lstm": LM_LSTM,
        "lstm_dropout": LM_LSTM_DROPOUT
    }

    optimizer_map = {
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW
    }

    model_class = model_map[args.model]
    model = model_class(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], out_dropout=out_dropout, emb_dropout=emb_dropout, n_layers=n_layers).to(device)
    model.apply(init_weights)

    if args.optimizer == "sgd":
        optimizer = optimizer_map[args.optimizer](model.parameters(), lr=lr)
    else:
        optimizer = optimizer_map[args.optimizer](model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    ######################################################################################################### train the model
  
    # If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            perplexities.append(ppl_dev)
            pbar.set_description(f"PPL: {ppl_dev:.2f} | LR: {lr:.5f} | hid_size: {hid_size} | emb_size: {emb_size} | batch_train: {batch_train} | batch_dev_test: {batch_dev_test}")

            if  ppl_dev < best_ppl: # The lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to(device)
                patience = patience
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)
    model_name = f"{args.model.upper()}_{args.optimizer.upper()}_PPL_{final_ppl:.2f}_LR_{lr}"

    save_training_results(model, best_model, final_ppl, lr, hid_size, emb_size, clip, 
                      n_epochs, patience, batch_train, batch_dev_test, 
                      sampled_epochs, losses_train, losses_dev, 
                      perplexities, plot_loss, plot_perplexity, model_name)