import torch
import torch.nn as nn
import torch.utils.data.dataloader as Dataloader
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class datasets:
    def __init__(self, root):
        texts = pd.read_csv(root)['sentence'].values.tolist()
        self.texts = list(map(lambda x: x.strip().lower(), texts))
        self.label = pd.read_csv(root)['label'].values.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        ids = tokenizer(self.texts[item], return_tensors='pt')['input_ids'].squeeze()
        return (ids, torch.tensor(self.label[item], dtype=torch.long))


class collate:
    def __init__(self):
        pass

    def __call__(self, batch):
        ids = [x[0] if x[0].shape else x[0].unsqueeze(0) for x in batch]
        label = torch.tensor([x[1] for x in batch])
        return (pad_sequence(ids, padding_value=0), label)


def get_data_iter(root, batch_size, shuffle):
    dataset = datasets(root=root)
    data = Dataloader.DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size,
                                 collate_fn=collate())
    return data
