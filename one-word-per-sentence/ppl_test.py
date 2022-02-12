import math
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import numpy

# Load pre-trained model (weights)
model_gpt = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
model_gpt.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')


def score(sentence):
    tokenize_input = tokenizer_gpt.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer_gpt.convert_tokens_to_ids(tokenize_input)]).cuda()
    loss = model_gpt(tensor_input, labels=tensor_input).loss
    return math.exp(loss)


for path in [
    './bert-base-uncased-gen/bert-only-1_sample.txt',
    './bert-base-uncased-gen/bert-only-10_sample.txt',
    './bert-base-uncased-gen/bert-only-20_sample.txt',
    './bert-base-uncased-gen/bert-only-pure_sample.txt',
    './bert-base-uncased-gen/bert-gibbs-1-turns-1_sample.txt',
    './bert-base-uncased-gen/bert-gibbs-10-turns-1_sample.txt',
    './bert-base-uncased-gen/bert-gibbs-20-turns-1_sample.txt',
    './bert-base-uncased-gen/bert-gibbs-pure-turns-1_sample.txt',
    './bert-base-uncased-gen/cover.txt',
]:

    with open(path, 'r', encoding='UTF-8') as f:
        texts = f.readlines()

    texts = tqdm(texts)
    scores = []
    for text in texts:
        s = score(text.strip())
        scores.append(s)
    PPL = numpy.mean(scores)
    with open('PPL.txt', 'a', encoding='UTF-8') as file:
        file.write(path + '\n')
        file.write(
            f'PPL  {PPL:.4f} \n')
