import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

class JointBert(BertPreTrainedModel):

    def __init__(self, config, out_slot, out_int, dropout):
        
        super(JointBert, self).__init__(config)
        self.bert_encoder = BertModel(config)
        self.slot_classifier = nn.Linear(config.hidden_size, out_slot)
        self.intent_classifier = nn.Linear(config.hidden_size, out_int)
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Dropout(0.1)

        
    def forward(self, input_ids, att_mask, token_type_ids):
        
        bert_outputs = self.bert_encoder(input_ids, attention_mask=att_mask, token_type_ids=token_type_ids)

        sequence_output = bert_outputs.last_hidden_state
        cls_output = bert_outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        cls_output = self.dropout(cls_output)

        # Slot predictions
        slots = self.slot_classifier(sequence_output)  # shape: (batch, seq_len, num_slot_labels)
        slots = slots.permute(0, 2, 1)  # shape: (batch, num_slot_labels, seq_len)

        # Intent prediction
        intent = self.intent_classifier(cls_output)  # shape: (batch, num_intent_labels)
        
        return slots, intent