import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from Datasets import get_data_iter
# from statistic.Datasets import get_data_iter
from Models import FCN, CNN, RNN, AE_RNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from test import fc
import os
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from torchtext.data import get_tokenizer

tokenizer = get_tokenizer("basic_english")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASKSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 设置随机数种子
setup_seed(2022)

for data_tmp in ['../bert-base-uncased-gen/bert-only-1_sample.txt',
                 '../bert-base-uncased-gen/bert-only-10_sample.txt',
                 '../bert-base-uncased-gen/bert-only-20_sample.txt',
                 '../bert-base-uncased-gen/bert-only-pure_sample.txt',
                 '../bert-base-uncased-gen/bert-gibbs-1-turns-1_sample.txt',
                 '../bert-base-uncased-gen/bert-gibbs-10-turns-1_sample.txt',
                 '../bert-base-uncased-gen/bert-gibbs-20-turns-1_sample.txt',
                 '../bert-base-uncased-gen/bert-gibbs-pure-turns-1_sample.txt'
                 ]:

    fc(data_tmp)


    def tokenize(text):
        return tokenizer(text)


    quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
    score = Field(sequential=False, use_vocab=False)

    fields = {"sentence": ("q", quote), "label": ("s", score)}

    train_data, val_data, test_data = TabularDataset.splits(
        path="data", train="train.csv", validation='val.csv', test="test.csv", format="csv", fields=fields
    )

    quote.build_vocab(train_data, max_size=20000, min_freq=1, vectors="glove.6B.300d")

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train_data, val_data, test_data), batch_size=64,
        sort_within_batch=True, sort_key=lambda x: len(x.q), device=device
    )

    VOCAB_SIZE = len(quote.vocab)
    print(VOCAB_SIZE)
    CLASSES = 2
    DEVICE = torch.device('cuda:0')
    HIDDEN_SIZE = 256
    EMBEDDING_DIM = 300  # transformer 128   rnn 256
    DROPOUT = 0.5
    HEDAS = 8
    MAX_LENGTH = 512
    RNN_NUM_LAYERS = 2
    Transformer_LAYERS = 2
    NUM_FILTERS = 100
    FILTER_SIZE = [1, 2, 3]  # [1,2,3] 100          [3,5,7] 128
    MODEL = 'rnn'

    if MODEL == 'cnn':
        model = CNN(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, num_filters=NUM_FILTERS,
                    filter_size=FILTER_SIZE, classes=CLASSES).to(DEVICE)
    elif MODEL == 'rnn':
        model = RNN(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE,
                    num_layers=RNN_NUM_LAYERS).to(DEVICE)
    elif MODEL == 'fcn':
        model = FCN(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, classes=CLASSES).to(DEVICE)
    elif MODEL == 'pt_transformer':
        model = torch.load('transformer.pth').to(DEVICE)
    elif MODEL == 'ae_rnn':
        model = AE_RNN(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE,
                       num_layers=RNN_NUM_LAYERS).to(DEVICE)

    EPOCHS = 100

    LR = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    classify_loss_fn = nn.CrossEntropyLoss()
    pretrained_embeddings = quote.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)


    def evaluate(model, data_iter):
        model.eval()
        predict = []
        correct = []
        with torch.no_grad():
            for data in data_iter:
                src = data.q.to(DEVICE)
                tar = data.s.to(DEVICE)
                output = model(src)[0]
                output = torch.argmax(output, dim=-1)
                predict.extend(output.cpu().numpy().tolist())
                correct.extend(tar.cpu().numpy().tolist())
        model.train()
        return accuracy_score(correct, predict), precision_score(correct, predict), recall_score(correct,
                                                                                                 predict), f1_score(
            correct, predict)


    best_val_acc = 0
    best_val_test_acc = 0
    best_val_test_p = 0
    best_val_test_r = 0
    loss_record = []

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.

        for idx, data in enumerate(train_iter):
            src = data.q.to(DEVICE)
            label = data.s.to(DEVICE)

            output = model(src)[0]

            loss = classify_loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            total_loss += loss.item()

        train_acc, train_p, train_r, train_f = evaluate(model, train_iter)
        val_acc, val_p, val_r, val_f = evaluate(model, val_iter)

        print(
            f'{epoch}/{EPOCHS}    loss {total_loss / len(train_iter):.4f}  best_val_test_acc {best_val_test_acc}  best_val_test_p {best_val_test_p} best_val_test_r {best_val_test_r}'
            f'train_acc, train_p, train_r {train_acc:.4f} {train_p:.4f} {train_r:.4f}  val_acc, val_p, val_r {val_acc:.4f} {val_p:.4f} {val_r:.4f}')
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            test_acc, test_p, test_r, test_f = evaluate(model, test_iter)

            best_val_test_acc = test_acc
            best_val_test_p = test_p
            best_val_test_r = test_r
            best_val_test_f = test_f
            print(f'test_acc  {test_acc:.4f}, test_p  {test_p:.4f}, test_r  {test_r:.4f} test_f  {test_f:.4f}')

        loss_record.append(total_loss / len(train_iter))

    f = './result/' + MODEL + '.txt'
    with open(f, 'a') as file:
        file.write(data_tmp + '\n')
        file.write(
            f'acc  {best_val_test_acc:.4f},  pre  {best_val_test_p:.4f},  recall  {best_val_test_r:.4f},  f1 {best_val_test_f:.4f} \n')
