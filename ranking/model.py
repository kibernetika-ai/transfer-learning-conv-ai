# author: Xiang Gao at Microsoft Research AI NLP Group
import os
import yaml

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config


class OptionInfer:
    def __init__(self, cuda=True):
        self.cuda = cuda


def get_model(path, eos_token, cuda=True):
    opt = OptionInfer(cuda)
    if os.path.isdir(path):
        model = JointScorer(opt, eos_token)
        model.load(path)
    elif path.endswith('.pth'):
        model = Scorer(opt, eos_token)
        model.load(path)
        if cuda:
            model.cuda()
    else:
        raise RuntimeError(f'Error path: {path}')
    return model


def predict(model, cxt, hyps, max_cxt_turn=None):
    # split into smaller batch to avoid OOM
    n = len(hyps)
    i0 = 0
    scores = []
    while i0 < n:
        i1 = min(i0 + 32, n)
        _scores = model.predict(cxt, hyps[i0: i1], max_cxt_turn=max_cxt_turn)
        scores.append(_scores)
        i0 = i1
    if isinstance(_scores, dict):
        d_scores = dict()
        for k in _scores:
            d_scores[k] = np.concatenate([_scores[k] for _scores in scores])
        return d_scores
    else:
        return np.concatenate(scores)


class ScorerBase(torch.nn.Module):
    def __init__(self, opt, eos_token):
        super().__init__()
        self.ix_EOS = 50256
        self.eos_token = eos_token
        self.ix_OMT = 986
        self.opt = opt
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def core(self, ids, l_ids, return_logits=False):
        # to be implemented in child class
        return 0

    def predict(self, cxt, hyps, max_cxt_turn=None):
        # cxt = str
        # hyps = list of str

        self.eval()
        cxt_turns = cxt.split(self.eos_token)
        if max_cxt_turn is not None:
            cxt_turns = cxt_turns[-min(max_cxt_turn, len(cxt_turns)):]
        ids_cxt = []
        for turn in cxt_turns:
            ids_cxt += self.tokenizer.encode(turn.strip()) + [self.ix_EOS]
        seqs = []
        lens = []
        for hyp in hyps:
            seq = ids_cxt + self.tokenizer.encode(hyp.strip())
            lens.append(len(seq))
            seqs.append(seq)
        max_len = max(lens)
        ids = []
        for seq in seqs:
            ids.append(seq + [self.ix_EOS] * (max_len - len(seq)))
        with torch.no_grad():
            ids = torch.LongTensor(ids)
            if self.opt.cuda:
                ids = ids.cuda()
            scores = self.core(ids, lens)
        if not isinstance(scores, dict):
            if self.opt.cuda:
                scores = scores.cpu()
            return scores.detach().numpy()

        for k in scores:
            if self.opt.cuda:
                scores[k] = scores[k].cpu()
            scores[k] = scores[k].detach().numpy()
        return scores

    def forward(self, batch):
        logits_pos = self.core(batch['ids_pos'], batch['len_pos'], return_logits=True)
        logits_neg = self.core(batch['ids_neg'], batch['len_neg'], return_logits=True)
        # softmax to get the `probability` to rank pos/neg correctly
        return torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))


class Scorer(ScorerBase):
    def __init__(self, opt, eos_token):
        super().__init__(opt, eos_token)
        n_embd = 1024
        config = GPT2Config(n_embd=n_embd, n_layer=24, n_head=16)
        self.transformer = GPT2Model(config)
        self.score = torch.nn.Linear(n_embd, 1, bias=False)

    def core(self, ids, l_ids, return_logits=False):
        n = ids.shape[0]
        attention_mask = torch.ones_like(ids)
        for i in range(n):
            attention_mask[i, l_ids[i]:] *= 0
        hidden_states, _ = self.transformer(ids, attention_mask=attention_mask)
        logits = self.score(hidden_states).squeeze(-1)
        logits = torch.stack([logits[i, l_ids[i] - 1] for i in range(n)])
        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)

    def load(self, path):
        # print('loading from ' + path)
        weights = torch.load(path)
        if path.endswith('.pkl'):
            # DialoGPT checkpoint
            weights['score.weight'] = weights['lm_head.decoder.weight'][self.ix_EOS: self.ix_EOS + 1, :]
            del weights['lm_head.decoder.weight']
        self.load_state_dict(weights, strict=False)


class JointScorer(ScorerBase):

    def core(self, ids, l_ids, return_logits=False):
        assert (not return_logits)
        scores = dict()
        for k in self.kk['prior'] + self.kk['cond']:
            scorer = getattr(self, 'scorer_%s' % k)
            scores[k] = scorer.core(ids, l_ids)

        def avg_score(kk):
            if not kk:
                return 1
            sum_score_wt = 0
            sum_wt = 0
            for k in kk:
                sum_score_wt = sum_score_wt + scores[k] * self.wt[k]
                sum_wt += self.wt[k]
            return sum_score_wt / sum_wt

        prior = avg_score(self.kk['prior'])
        cond = avg_score(self.kk['cond'])
        scores['final'] = prior * cond
        return scores

    def load(self, model_dir):
        with open(os.path.join(model_dir, 'ensemble.yml'), 'r') as stream:
            config = yaml.safe_load(stream)
        # print(config)

        paths = dict()
        self.wt = dict()
        self.kk = dict()
        for prefix in ['prior', 'cond']:
            self.kk[prefix] = []
            for d in config[prefix]:
                k = d['name']
                self.kk[prefix].append(k)
                self.wt[k] = d['wt']
                paths[k] = d['path']

        for k in paths:
            path = paths[k]
            # print('setting up model `%s`' % k)
            scorer = Scorer(OptionInfer(cuda=self.opt.cuda), eos_token=self.eos_token)
            scorer.load(path)
            if self.opt.cuda:
                scorer.cuda()
            setattr(self, 'scorer_%s' % k, scorer)
