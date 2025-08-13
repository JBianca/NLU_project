import torch
import torch.nn as nn

# LM_LSTM with weight tying
# class LM_LSTM_WT(nn.Module):
#     def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
#                  emb_dropout=0.1, n_layers=1):
#         super(LM_LSTM_WT, self).__init__()

#         self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
#         self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
#         self.pad_token = pad_index
#         self.output = nn.Linear(hidden_size, output_size)

#         if emb_size == hidden_size:
#             self.output.weight = self.embedding.weight  # Weight tying 
#         else:
#             print("Weight tying not applicable due to difference hidden size and embedding size")
    
#     def forward(self, input_sequence):
#         emb = self.embedding(input_sequence)
#         lstm_out, _ = self.lstm(emb)
#         output = self.output(lstm_out).permute(0, 2, 1)
#         return output


class VariationalDropout(nn.Module):
    
    def __init__(self, dropout=0.5):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x


class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0,
                 out_dropout=0.1, emb_dropout=0.1, n_layers=1, weight_tying=False, variational_dropout=False):
        super(LM_LSTM, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.drop1 = VariationalDropout(dropout=emb_dropout) if variational_dropout else nn.Dropout(p=emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, dropout=out_dropout if n_layers > 1 else 0 )
        self.pad_token = pad_index
        self.drop2 = VariationalDropout(dropout=out_dropout) if variational_dropout else nn.Dropout(p=emb_dropout)
        self.output = nn.Linear(hidden_size, output_size)

        if weight_tying:
            assert hidden_size == emb_size, "Weight tying not applicable due to difference hidden size and embedding size"
            self.output.weight = self.embedding.weight  # Weight tying 

    def forward(self, input_sequence):
        
        emb = self.embedding(input_sequence) 
        drop1 = self.drop1(emb)  # embedding and apply dropout
        lstm_out, _ = self.lstm(drop1)
        drop2 = self.drop2(lstm_out) # lstm output and apply dropout
        output = self.output(drop2).permute(0, 2, 1)
        return output