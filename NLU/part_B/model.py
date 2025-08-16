import torch.nn as nn
from transformers import BertModel

class JointBert(nn.Module):
    def __init__(self, hidden_dim, slot_labels, intent_labels, dropout=0.1):
        super(JointBert, self).__init__()

        # Load pretrained BERT encoder 
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        # Dropout for regularization to reduce overfitting
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer for slot filling (token-level classification)
        self.slot_classifier = nn.Linear(hidden_dim, slot_labels)
        
        # Linear layer for intent classification (sequence-level classification)
        self.intent_classifier = nn.Linear(hidden_dim, intent_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Args:
            input_ids: Tensor of token IDs, shape (batch_size, seq_len)
            attention_mask: Tensor mask (1 = real token, 0 = padding), shape (batch_size, seq_len)
            token_type_ids: Tensor segment IDs for sentence-pair tasks, shape (batch_size, seq_len)
        
        Returns:
            slots: Slot label logits, shape (batch_size, slot_labels, seq_len)
            intent: Intent label logits, shape (batch_size, intent_labels)
        """

        # BERT forward pass
        bert_outputs = self.bert_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Token embeddings from last hidden state
        token_embeddings = self.dropout(bert_outputs.last_hidden_state)             # shape: (batch, seq_len, hidden_dim)
        
        # [CLS] token embedding for sequence-level classification
        cls_embedding = self.dropout(bert_outputs.pooler_output)                    # shape: (batch, hidden_dim)
        
        # Slot label predictions for each token
        slots = self.slot_classifier(token_embeddings)                              # shape: (batch, seq_len, slot_labels)
        slots = slots.permute(0, 2, 1)                                              # shape: (batch, slot_labels, seq_len)

        # Intent label prediction for the whole utterance
        intent = self.intent_classifier(cls_embedding)                              # shape: (batch, intent_labels)
        
        return slots, intent