import torch
import torch.nn as nn

class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        # Create the same dropout mask for all timesteps
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x

# LSTM with weight tying and variational dropout as boolean variable setting
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0,
                 out_dropout=0.1, emb_dropout=0.1, n_layers=1,
                 weight_tying=False, variational_dropout=False):
        super(LM_LSTM, self).__init__()

        self.variational_dropout = variational_dropout
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.drop1 = VariationalDropout(dropout=emb_dropout) if variational_dropout else nn.Dropout(p=emb_dropout)

        # Conditional LSTM dropout
        lstm_dropout = 0 if variational_dropout else (out_dropout if n_layers > 1 else 0)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, dropout=lstm_dropout)

        self.drop2 = VariationalDropout(dropout=out_dropout) if variational_dropout else nn.Dropout(p=out_dropout)
        self.output = nn.Linear(hidden_size, output_size)

        # Weight tying
        if weight_tying:
            assert hidden_size == emb_size, "Weight tying requires hidden_size == emb_size"
            self.output.weight = self.embedding.weight

        print(f"Variational Dropout: {'ON' if variational_dropout else 'OFF'}")

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop1 = self.drop1(emb)
        lstm_out, _ = self.lstm(drop1)
        drop2 = self.drop2(lstm_out)
        output = self.output(drop2).permute(0, 2, 1)
        return output