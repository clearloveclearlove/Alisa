import torch
import numpy as np
import torch.nn.functional as F
import random
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


def bert_gibbs(given_words, length, pos, turns, sample='pure'):
    mask_sentence = [mask] * length
    pos.sort()
    for i, j in zip(pos, given_words):
        mask_sentence[i] = tokenizer.convert_tokens_to_ids(j)
    fill_pos = list(set(list(range(length))) - set(pos))

    for _ in range(turns):

        for p in fill_pos:
            ids = torch.tensor([start] + mask_sentence + [end]).unsqueeze(0).cuda()
            logits = model(ids).logits
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

    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(mask_sentence))


def bert_only(given_words, length, pos, sample='pure'):
    mask_sentence = [mask] * length
    pos.sort()
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

    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(mask_sentence))


if __name__ == '__main__':
    pos_key = [3, 9]
    # given_words = ['go', 'ahead']
    given_words = ['come', 'on']

    setup_seed(7)
    length = random.randint(10, 20)

    bert_gibbs_text = bert_gibbs(given_words, length, pos_key, turns=1, sample=20)
    bert_only_text = bert_only(given_words, length, pos_key, sample=20)
    print('given_words:  ', ' '.join(given_words))
    print('position key:  ', ' '.join([str(p) for p in pos_key]))
    print('bert-gibbs-steg:  ', bert_gibbs_text)
    print('bert-only-steg:  ', bert_only_text)
    print('========================================================================')
