# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import torch.nn as nn
import math
import os
import matplotlib.pyplot as plt

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

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

# Plot and save the training and validation loss
def plot_loss(epochs, train_loss, dev_loss, save_path, model_name=""):
    plt.title(f'Training and Validation Loss {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(epochs, dev_loss, label='Validation Loss', marker='s')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot and save the validation perplexity
def plot_perplexity(epochs, perplexities, save_path, model_name=""):
    plt.title(f'Validation Perplexity {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, perplexities, label='Validation Perplexity', color='tab:orange', marker='^')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def save_training_results(model, best_model, final_ppl, lr, hid_size, emb_size, clip, 
                           n_epochs, sampled_epochs, patience, patience_AvSGD, batch_train, batch_dev_test,
                           losses_train, losses_dev, perplexities, plot_loss, plot_perplexity, model_name,
                           out_dropout, emb_dropout, n_layers, weight_tying, variational_dropout, monotone,
                           asgd_triggered_epoch, use_avsgd, results_dir="results"):

    result_path = os.path.join(results_dir, model_name)
    os.makedirs(result_path, exist_ok=True)

    # Save training summary
    results_file = os.path.join(result_path, "results.txt")
    with open(results_file, "w") as f:
        f.write(f"Model: {model_name}\n\n")
        f.write(f"hid_size={hid_size}\n")
        f.write(f"emb_size={emb_size}\n")
        f.write(f"lr={lr}\n")
        f.write(f"clip={clip}\n")
        f.write(f"n_epochs={n_epochs}\n")
        f.write(f"out_dropout={out_dropout}\n")
        f.write(f"emb_dropout={emb_dropout}\n")
        f.write(f"n_layers={n_layers}\n")
        f.write(f"weight_tying={weight_tying}\n")
        f.write(f"variational_dropout={variational_dropout}\n")
        f.write(f"monotone={monotone}\n")        
        f.write(f"patience={patience}\n")
        f.write(f"patience_AvSGD={patience_AvSGD}\n")
        f.write(f"batch_train={batch_train}\n")
        f.write(f"batch_dev_test={batch_dev_test}\n")
        f.write(f"test_ppl={final_ppl:.2f}\n")
        f.write(f"use_avsgd={use_avsgd}\n")
        if asgd_triggered_epoch is not None:
            f.write(f"ASGD triggered at epoch: {asgd_triggered_epoch}\n")
        else:
            f.write("ASGD was never triggered\n")

    # Save loss plot
    loss_path = os.path.join(result_path, "loss_plot.png")
    plot_loss(sampled_epochs, losses_train, losses_dev, loss_path, model_name=model_name)

    # Save perplexity plot
    ppl_path = os.path.join(result_path, "perplexity_plot.png")
    plot_perplexity(sampled_epochs, perplexities, ppl_path, model_name=model_name)

    # Save model
    model_path = os.path.join(result_path, f"{model_name}.pt")
    torch.save(best_model.state_dict(), model_path)