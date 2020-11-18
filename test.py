import os
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# import huggingface transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values,
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'),
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    return logits


gpt2_small_config = GPT2Config()
gpt2_medium_config = GPT2Config(n_ctx=1024, n_embd=1024, n_layer=24, n_head=16)
gpt2_large_config = GPT2Config(n_ctx=1024, n_embd=1280, n_layer=36, n_head=20)
# load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# load the model
model_size = "medium"

if model_size == "small":
    model = GPT2LMHeadModel(gpt2_small_config)
    model.load_state_dict(torch.load("small_ft.pkl"), strict=False)
    # model.load_state_dict(torch.load("medium_ft.pkl"), strict=False)
elif model_size == "medium":
    model = GPT2LMHeadModel(gpt2_medium_config)
    model.load_state_dict(torch.load("medium_ft.pkl"), strict=False)
    # model.load_state_dict(torch.load("medium-ft/pytorch_model.bin"), strict=False)
elif model_size == "large":
    model = GPT2LMHeadModel(gpt2_large_config)
    model.load_state_dict(torch.load("large_ft.pkl"), strict=False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# beg huggingface not to change this anymore
model.lm_head.weight.data = model.transformer.wte.weight.data

eos = [tokenizer.encoder["<|endoftext|>"]]

past = None
temperature = 0.9
top_k = -1
top_p = 0.9

model.eval()
prev_input = None

while True:
    with torch.no_grad():
        # input and update B's utterance
        user = input("User: ")

        if user == "quit":
            "stop talking!"
            break

        input_ids = tokenizer.encode(user, return_tensors='pt').to(device)
        output = model.generate(
            input_ids,
            do_sample=True,
            max_length=50,
            top_p=0.92,
            top_k=50
        )
        print(f'Bot: {tokenizer.decode(output[0], skip_special_tokens=True)}')
        # user = tokenizer.encode(user)
        # prev_input = user
        # prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)
        # _, past = model(prev_input, past=past)
        #
        # prev_input = torch.LongTensor([eos]).to(device)
        #
        # sent = []
        # for i in range(500):
        #     logits, past = model(prev_input, past=past)
        #     logits = logits[:, -1, :] / temperature
        #     logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        #
        #     probs = torch.softmax(logits, dim=-1)
        #
        #     prev_input = torch.multinomial(probs, num_samples=1)
        #     prev_word = prev_input.item()
        #
        #     if prev_word == eos[0]:
        #         break
        #     sent.append(prev_word)

        # print("Bot:", tokenizer.decode(sent))
        # prev_input = torch.LongTensor([eos]).to(device)
        # _, past = model(prev_input, past=past)
