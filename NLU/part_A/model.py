import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, use_drop=False, use_bidirectional=False, dropout=0.5):
        super(ModelIAS, self).__init__()

        self.use_drop = use_drop
        self.use_bidirectional = use_bidirectional
        self.num_directions = 2 if use_bidirectional else 1

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=use_bidirectional, batch_first=True)    
        
        # Dropout layer How/Where do we apply it?
        if use_drop:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        
        self.slot_out = nn.Linear(hid_size * self.num_directions, out_slot)
        self.intent_out = nn.Linear(hid_size * self.num_directions, out_int)
         
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        utt_emb = self.dropout(utt_emb)
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)

        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout on sequence output before slot classification
        utt_encoded = self.dropout(utt_encoded)
        
        # Get last hidden state for intent classification
        if self.use_bidirectional:
            last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
        else:
            last_hidden = last_hidden[-1]
        
        last_hidden = self.dropout(last_hidden)

        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent