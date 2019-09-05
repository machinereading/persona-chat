import torch
from torch.utils.data import Dataset
import random
from utils.vocab import CustomVocab

class CustomDataset(Dataset):
    @staticmethod
    def parse_data(path):
        with open(path, 'r', encoding='utf-8') as file:
            data = []
            for line in file.readlines():
                line = line.strip()

                if len(line) == 0:
                    continue

                space_idx = line.find(' ')
                if space_idx == -1:
                    dialog_idx = int(line)
                else:
                    dialog_idx = int(line[:space_idx])

                if int(dialog_idx) == 1:
                    data.append({'persona_info': [], 'dialog': []})

                dialog_line = line[space_idx + 1:].split('\t')
                dialog_line = [l.strip() for l in dialog_line]

                if dialog_line[0].startswith('your persona:'):
                    persona_info = dialog_line[0].replace('your persona: ', '')
                    data[-1]['persona_info'].append(persona_info)

                elif len(dialog_line) > 1:
                    data[-1]['dialog'].append(dialog_line[0])
                    data[-1]['dialog'].append(dialog_line[1])

            return data

    @staticmethod
    def make_dataset(data, vocab, max_lengths):
        dataset = []
        for chat in data:
            persona_info = [vocab.string2ids(s) for s in chat['persona_info']]
            persona_info = [[(w, vocab.p_id) for w in s + [vocab.eos_id]] for s in persona_info]

            dialog = [vocab.string2ids(s) for s in chat['dialog']]
            dialog = [[(w, vocab.s1_id) for w in s + [vocab.eos_id]] if i%2==0 \
                      else [(w, vocab.s2_id) for w in s + [vocab.eos_id]] for i, s in enumerate(dialog)]

            if len(dialog) % 2 == 1:
                dialog = dialog[:-1]
           
            dataset.append((persona_info, dialog))

        return dataset

    def __init__(self, paths, vocab, max_lengths=2048, min_infos=2):
        assert min_infos > 0             

        if isinstance(paths, str):
            paths = [paths]
        
        self.vocab = vocab
        self.max_lengths = max_lengths
        self.min_infos = min_infos

        parsed_data = sum([CustomDataset.parse_data(path) for path in paths], [])
        self.data = CustomDataset.make_dataset(parsed_data, vocab, max_lengths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        persona_info, dialog = self.data[idx]
        x = x = sum(persona_info, []) + sum(dialog[:-1], []) + [(self.vocab.sentinel_id, self.vocab.sentinel_id)]
        y = [d[0] for d in dialog[-1]]
        
        # Pointers for MemNN
        ptrs = []
        for word in y:
            ptr = -1
            for idx in range(len(x)-1):
                if x[idx][0] == word:
                    ptr = idx
            if ptr < 0:
                ptr = idx + 1
                
            ptrs.append(ptr)
            
        # Plain sequences
        x_plain = [self.vocab.id2token[id1] for id1, id2 in x]
        y_plain = [self.vocab.id2token[idx] for idx in y]

        return x, y, ptrs, x_plain, y_plain
    
class Collator:
    def __init__(self, device, pad_id):
        self.device = device
        self.pad_id = pad_id
    
    def __call__(self, data):
        def merge(seqs, paired=False):
            lengths = [len(seq) for seq in seqs]
            
            if paired:
                padded_seqs = torch.zeros(len(seqs), max(lengths), 2,
                                          dtype=torch.long, device=self.device).fill_(self.pad_id)
                for i, seq in enumerate(seqs):
                    end = lengths[i]
                    for j in range(end):
                        padded_seqs[i, j, 0] = seq[j][0]
                        padded_seqs[i, j, 1] = seq[j][1]
            else:
                padded_seqs = torch.zeros(len(seqs), max(lengths),
                                          dtype=torch.long, device=self.device).fill_(self.pad_id)
                for i, seq in enumerate(seqs):
                    end = lengths[i]
                    for j in range(end):
                        padded_seqs[i, j] = seq[j]
                        padded_seqs[i, j] = seq[j]

            return padded_seqs, lengths

        stories, responses, pointers, src_plain, trg_plain = zip(*data)
        src_seqs, src_lengths = merge(stories, True)
        trg_seqs, trg_lengths = merge(responses)
        ind_seqs, _ = merge(pointers)
        
        src_seqs = src_seqs.transpose(0, 1)
        trg_seqs = trg_seqs.transpose(0, 1)
        ind_seqs = ind_seqs.transpose(0, 1)
        
        return src_seqs, src_lengths, trg_seqs, trg_lengths, ind_seqs, src_plain, trg_plain
    

def preprocess(vocab_path, codes_path, train_datasets, valid_datasets, test_datasets, batch_size, device):
    vocab = CustomVocab.from_files(vocab_path, codes_path)
    
    train_dataset = CustomDataset(train_datasets, vocab)
    valid_dataset = CustomDataset(valid_datasets, vocab)
    test_dataset = CustomDataset(test_datasets, vocab)
    
    max_r = 0
    for data in train_dataset:
        if max_r <= len(data[1]):
            max_r = len(data[1])
    for data in valid_dataset:
        if max_r <= len(data[1]):
            max_r = len(data[1])
    
    collator = Collator(device, vocab.pad_id)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=collator)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False,
                                               collate_fn=collator)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=collator)
    
    return vocab, train_loader, valid_loader, test_loader, max_r
