from conll import evaluate
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def plot_losses(epochs, train_loss, dev_loss, save_path, title="Training and Validation Loss"):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(epochs, dev_loss, label='Validation Loss', marker='s')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def build_model_name(lr, slot_f1s, intent_acc, bidirectional=False, dropout=False):
    name_parts = ["IAS"]
    
    if bidirectional:
        name_parts.append("BI")
    if dropout:
        name_parts.append("DROP")
    
    name_parts.append(f"lr{lr}")
    name_parts.append(f"F1_{round(np.mean(slot_f1s), 3)}")
    name_parts.append(f"INTACC_{round(np.mean(intent_acc), 3)}")
    
    return "_".join(name_parts)

def save_experiment_results(model, optimizer, n_epochs, lr, hid_size, emb_size, slot_f1s, intent_acc,
                            losses_train, losses_dev, runs, dropout, bidirectional, patience, model_name, results_dir="results"):
    
    result_path = os.path.join(results_dir, model_name)
    os.makedirs(result_path, exist_ok=True)

    # Save results summary
    results_file = os.path.join(result_path, f"results.txt")
    with open(results_file, "w") as f:
        f.write(f"Epoch: {n_epochs}\n")
        f.write(f"LR: {lr}\n")
        f.write(f"Hidden Size: {hid_size}\n")
        f.write(f"Embedding Size: {emb_size}\n")
        f.write("Optimizer: Adam\n")
        f.write(f"Slot F1: {slot_f1s.mean():.3f} ± {slot_f1s.std():.3f}\n")
        f.write(f"Intent Acc: {intent_acc.mean():.3f} ± {intent_acc.std():.3f}\n")
        f.write(f"Runs: {runs}\n")
        f.write(f"Dropout: {dropout}\n")
        f.write(f"Bidirectional: {bidirectional}\n")
        f.write(f"Patience: {patience}\n")

    print(f"Slot F1: {slot_f1s.mean():.3f} ± {slot_f1s.std():.3f}")
    print(f"Intent Acc: {intent_acc.mean():.3f} ± {intent_acc.std():.3f}")

    # Save loss plot
    plot_file = os.path.join(result_path, f"loss.png")
    plot_losses(range(1, len(losses_train) + 1), losses_train, losses_dev, plot_file)

    # Save model
    model_path = os.path.join(result_path, f"{model_name}_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)