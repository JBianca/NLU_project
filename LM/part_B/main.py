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
import torch.optim as optim

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

    batch_train = 64
    batch_dev_test = 128

    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=batch_train, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_dev_test, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=batch_dev_test, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    ######################################################################################################### parameters configuration
    
    hid_size = 500
    emb_size = 500
    vocab_len = len(lang.word2id)
    lr = 10 # [0.05, 0.01] for SGD; [0.001] for Adam
    clip = 5 # Clip the gradient
    monotone = 5
    n_epochs = 60
    patience = 3
    patience_AvSGD = 5
    
    out_dropout = 0.2
    emb_dropout = 0.6
    n_layers = 1
    weight_tying = True
    variational_dropout = True
    use_avsgd = True

    ##############################################################

    asgd_active = False
    asgd_triggered_epoch = None
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    perplexities = [] # add pp vector for plot
    best_val_loss = []
    best_ppl = math.inf
    best_model = None
    best_loss = math.inf
    pbar = tqdm(range(1,n_epochs+1))

    ######################################################################################################### model

    model = LM_LSTM(
        emb_size=emb_size,
        hidden_size=hid_size,
        output_size=vocab_len,
        pad_index=lang.word2id["<pad>"],
        out_dropout=out_dropout,
        emb_dropout=emb_dropout,
        n_layers=n_layers,
        weight_tying=weight_tying,
        variational_dropout=variational_dropout
    ).to(device)
    init_weights(model)
   
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)

    ######################################################################################################### train the model

    
    # If the PPL is too high try to change the learning rate
    for epoch in pbar:
        # Training loop for the current epoch
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip=clip)
        
        # Every epoch, perform validation
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())  # Store the average loss for this epoch

            # Check if optimizer is using ASGD
            if 't0' in optimizer.param_groups[0]:  # ASGD is being used
                # Temporarily store the model's weights
                temp_weights = {}
                for param in model.parameters():
                    temp_weights[param] = param.data.clone()
                    param.data = optimizer.state[param]['ax'].clone()  # Use ASGD's averaged weights for validation

                # Evaluate the model on the validation set
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                perplexities.append(ppl_dev)
                losses_dev.append(np.asarray(loss_dev).mean())

                # Restore the original model weights
                for param in model.parameters():
                    param.data = temp_weights[param].clone()

                # Check if the current perplexity is the best so far
                if ppl_dev < best_ppl:
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to(device)  # Save a copy of the best model
                    patience = 3  # Reset patience because the model improved
                else:
                    patience -= 1  # Decrease patience if there's no improvement

                # Early stopping if patience runs out
                if patience <= 0:
                    print("Early stopping triggered: No improvement in perplexity for the last few epochs.")
                    break  # Exit training loop early if patience is exhausted

            else:  # ASGD is not being used, continue with normal validation
                # Evaluate the model on the validation set
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                perplexities.append(ppl_dev)
                losses_dev.append(np.asarray(loss_dev).mean())

                # Check if it's time to switch to ASGD optimizer
                if use_avsgd and 't0' not in optimizer.param_groups[0] and len(best_val_loss) > monotone and loss_dev > min(best_val_loss[:-monotone] or [float('inf')]):
                    print('Switch to ASGD')
                    patience = patience_AvSGD
                    lr = lr*0.4
                    optimizer.param_groups[0]['lr'] = lr
                    optimizer = torch.optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=1.2e-6,)
                    asgd_triggered_epoch = epoch  # store when ASGD started
                    asgd_active = True

                # Update the best validation loss if we have a new best loss
                if loss_dev < best_loss:
                    best_loss = loss_dev

                # Track the best validation loss for later plotting
                best_val_loss.append(loss_dev)

            # Update progress bar description with current metrics
            status = "ASGD" if asgd_active else "SGD"
            pbar.set_description(f"PPL: {ppl_dev:.2f} | LR: {lr:.3f} | hid_s: {hid_size} | emb_s: {emb_size} | o_drop: {out_dropout} | e_drop: {emb_dropout} | wt: {weight_tying} | vd: {variational_dropout} | {status}")


            # Check if the current perplexity is the best so far
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to(device)  # Save a copy of the best model
                patience = 5  # Reset patience because the model improved
            else:
                patience -= 1  # Decrease patience if there's no improvement

            # Early stopping if patience runs out
            if patience <= 0:
                print("Early stopping triggered: No improvement in perplexity for the last few epochs.")
                break  # Exit training loop early if patience is exhausted
        
        if scheduler is not None:
                try:
                    scheduler.step()
                except TypeError:
                    scheduler.step(ppl_dev)  

    if best_model is None:
            best_model = model
    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)

    # Build tag list based on features
    tags = []
    if weight_tying:
        tags.append("wt")
    if variational_dropout:
        tags.append("vd")
    if asgd_active:  # use actual ASGD status
        tags.append("asgd")

    # Join tags with underscores if any
    tag_str = "_" + "_".join(tags) if tags else ""

    # Build model name
    model_name = f"{best_model.__class__.__name__}{tag_str}_PPL_{final_ppl:.2f}_{emb_size}_{out_dropout}_{emb_dropout}"
    save_training_results(model, best_model, final_ppl, lr, hid_size, emb_size, clip, 
                           n_epochs, sampled_epochs,  patience, patience_AvSGD, batch_train, batch_dev_test,
                           losses_train, losses_dev, perplexities, plot_loss, plot_perplexity, model_name,
                           out_dropout, emb_dropout, n_layers, weight_tying, variational_dropout, monotone,
                           asgd_triggered_epoch, use_avsgd)