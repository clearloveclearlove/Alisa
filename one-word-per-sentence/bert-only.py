import torch
import numpy as np
import torch.nn.functional as F
import random
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased').cuda()
model.eval()
start = tokenizer.cls_token_id
end = tokenizer.sep_token_id
mask = tokenizer.mask_token_id

for sample in [1, 10, 20, 'pure']:

    def bert_generate(given_words, length, pos, sample='pure'):
        mask_sentence = [mask] * length

        for i, j in zip(pos, given_words):
            mask_sentence[i] = tokenizer.convert_tokens_to_ids(j)

        fill_pos = list(set(list(range(length))) - set(pos))
        ids = torch.tensor([start] + mask_sentence + [end]).unsqueeze(0).cuda()
        logits = model(ids).logits

        for p in fill_pos:
            logit = logits[0][p + 1]

            # pure sample
            if sample == 'pure':
                dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(logit, dim=-1))
                sample_id = dist.sample().item()

            # top k
            if isinstance(sample, int):
                kth_vals, kth_idx = logit.topk(sample, dim=-1)
                dist = torch.distributions.categorical.Categorical(logits=kth_vals)
                idx = kth_idx.gather(dim=-1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
                sample_id = idx.item()

            mask_sentence[p] = sample_id

            if mask not in mask_sentence:
                return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(mask_sentence))


    with open('book.txt', 'r') as f:
        texts = f.readlines()

    texts = texts[:10000]
    texts = tqdm(texts)
    cover_texts = []
    stega_texts = []
    given_texts = []

    lengths = []
    for i, text in enumerate(texts):
        setup_seed(i)
        text = text.strip()
        cover_texts.append(text)
        tokenize_text = tokenizer.tokenize(text)
        length = len(tokenize_text)
        lengths.append(length)
        pos = random.sample(list(range(length)),1)
        pos.sort()
        given_words = [tokenize_text[p] for p in pos]
        given_texts.append(' '.join(given_words))
        stega_text = bert_generate(given_words, length, pos, sample=sample)
        stega_texts.append(stega_text)
        print('cover:  ', text)
        print('given_words:  ', ' '.join(given_words))
        print('steg:  ', stega_text)
        print('========================================================================')

    print(np.mean(lengths))
    with open('./bert-base-uncased-gen/bert-only-' + str(sample) + '_sample.txt', 'w', encoding='UTF-8') as f:
        f.write('\n'.join(stega_texts))
    with open('./bert-base-uncased-gen/given.txt', 'w', encoding='UTF-8') as f:
        f.write('\n'.join(given_texts))
    with open('./bert-base-uncased-gen/cover.txt', 'w', encoding='UTF-8') as f:
        f.write('\n'.join(cover_texts))
