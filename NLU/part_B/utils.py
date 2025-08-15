# Add functions or classes used for data loading and preprocessing
import json
from collections import Counter
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def get_splits(train_data, test_data, split_ratio=0.1):
    '''
        input: train_data, test_data, split_ratio
        output: train_set, dev_set
    '''

    intents = [x['intent'] for x in train_data] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(train_data[id_y])
            labels.append(y)
        else:
            mini_train.append(train_data[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=split_ratio, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_data]
    return train_raw, dev_raw, test_data, y_train, y_dev, y_test 

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': 0}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = 0
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    
PAD_TOKEN = 0   

class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, tokenizer, lang, unk='unk'):
        
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids, self.slots_ids, self.attention_mask, self.token_type_id = self.mapping_seq(
            self.utterances, self.slots, tokenizer, lang.slot2id
        )
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        sample = {
        'utterance': torch.tensor(self.utt_ids[idx], dtype=torch.long),
        'slots': torch.tensor(self.slots_ids[idx], dtype=torch.long),
        'intent': self.intent_ids[idx],
        'attention': torch.tensor(self.attention_mask[idx], dtype=torch.long),
        'token_type_id': torch.tensor(self.token_type_id[idx], dtype=torch.long)
        }
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, utterances, slots, tokenizer, mapper_slot): 
        utt_tokenized = []
        slots_tokenized = []
        att_mask_list = []
        token_type_list = []

        for utterance, slot in zip(utterances, slots):
            seq_tokens = []
            slot_tokens = []
            attention = []
            token_type_id = []

            for word, element in zip(utterance.split(), slot.split()):
                # Tokenize word
                word_tokens = tokenizer.encode(word, add_special_tokens=False)
                num_tokens = len(word_tokens)
                
                # Extend token lists
                seq_tokens.extend(word_tokens)
                slot_tokens.extend([mapper_slot[element]] + [mapper_slot['pad']] * (num_tokens - 1))
                
                # Extend attention and token type
                attention.extend([1] * num_tokens)
                token_type_id.extend([0] * num_tokens)

            utt_tokenized.append(seq_tokens)
            slots_tokenized.append(slot_tokens)
            att_mask_list.append(attention)
            token_type_list.append(token_type_id)
        
        return utt_tokenized, slots_tokenized, att_mask_list, token_type_list

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(0)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        return padded_seqs, lengths
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    attention, _ = merge(new_item['attention'])
    token_type_id, _ = merge(new_item["token_type_id"])
    
    
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    attention = attention.to(device)
    token_type_id = token_type_id.to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    new_item["attentions"] = attention
    new_item["token_type_ids"] = token_type_id
    
    return new_item