import torch
import torch.nn as nn

# RNN network - taken from lab
class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output 
    
# LSTM network
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output

# LSTM with two dropout layers:
#    - one after the embedding layer, 
#    - one before the last linear layer
class LM_LSTM_DROPOUT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                    emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_DROPOUT, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index) # Embedding layer
        self.emb_dropout = nn.Dropout(emb_dropout)  # Dropout layer after embedding layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True) # LSTM layer
        self.pad_token = pad_index
        self.output_dropout = nn.Dropout(out_dropout)  # Dropout layer before last linear layer
        self.output = nn.Linear(hidden_size, output_size) # Final layer

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence) # Embedding layer
        drop_after_emb = self.emb_dropout(emb) # Dropout after embedding
        lstm_out, _ = self.lstm(drop_after_emb) # LSTM layer 
        drop_before_last_layer = self.output_dropout(lstm_out) # Dropout before linear layer
        output = self.output(drop_before_last_layer).permute(0, 2, 1) # Final output
        return output