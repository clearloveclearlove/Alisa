import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy
from transformers import GPT2Model, GPT2Tokenizer


class Bert_fc(nn.Module):
    def __init__(self):
        super(Bert_fc, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        x = x.permute(1, 0)
        x = self.bert(x)
        x = self.fc(x[1])

        return x


class GPT2_fc(nn.Module):
    def __init__(self):
        super(GPT2_fc, self).__init__()
        self.gpt = GPT2Model.from_pretrained('gpt2')
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        x = x.permute(1, 0)
        text_lengths = torch.sum(x != 0, dim=1, dtype=torch.long) - 1
        mask = torch.zeros_like(x, dtype=torch.long).masked_fill(x != 0, value=1).to(x.device)
        x = self.gpt(input_ids=x, attention_mask=mask)[0]
        x = self.fc(x)

        out = []
        for t, l in zip(x, text_lengths):
            out.append(t[l, :].unsqueeze(0))

        out = torch.cat(out, dim=0)

        return out


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.5):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.lm_fc = nn.Linear(hidden_size, vocab_size)
        self.classify_fc = nn.Linear(2 * hidden_size, 2)
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size

    def forward(self, x, hidden=None, cell=None):
        # x -->(length,N)
        text_lengths = torch.sum(x != 0, dim=0).cpu().numpy().tolist()

        x = self.dropout(self.embedding(x))

        x = nn.utils.rnn.pack_padded_sequence(x, text_lengths, enforce_sorted=False)

        # x -->(length,N,embedding_dim)
        if hidden is not None:
            x, (hidden, cell) = self.rnn(x, (hidden, cell))
        else:
            x, (hidden, cell) = self.rnn(x)
        # x -->(length,N,hidden_size)

        # unpack sequence
        x, output_lengths = nn.utils.rnn.pad_packed_sequence(x)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hiddens = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.classify_fc(hiddens), self.lm_fc(x).reshape(-1, self.vocab_size), (hidden, cell)
