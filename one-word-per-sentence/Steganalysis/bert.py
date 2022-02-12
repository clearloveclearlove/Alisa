import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from Datasets import get_data_iter
from Models import RNN, Bert_fc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from test import fc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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
    from transformers import BertTokenizer

    vocab = BertTokenizer.from_pretrained('bert-base-uncased')

    train_iter = get_data_iter(root='./data/train.csv', batch_size=64, shuffle=True)
    val_iter = get_data_iter(root='./data/val.csv', batch_size=64, shuffle=True)
    test_iter = get_data_iter(root='./data/test.csv', batch_size=64, shuffle=True)

    CLASSES = 2
    DEVICE = torch.device('cuda:0')
    HIDDEN_SIZE = 256
    EMBEDDING_DIM = 300
    RNN_NUM_LAYERS = 2
    MODEL = 'bert'

    if MODEL == 'rnn':
        model = RNN(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE,
                    num_layers=RNN_NUM_LAYERS).to(DEVICE)
    elif MODEL == 'bert':
        model = Bert_fc().to(DEVICE)

    EPOCHS = 30
    LR = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    classify_loss_fn = nn.CrossEntropyLoss()


    def evaluate(model, data_iter):
        model.eval()
        predict = []
        correct = []
        with torch.no_grad():
            for data in data_iter:
                src = data[0].to(DEVICE)
                tar = data[1].to(DEVICE)
                output = model(src)
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
            src = data[0].to(DEVICE)
            label = data[1].to(DEVICE)

            output = model(src)

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
