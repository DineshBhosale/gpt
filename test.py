import torch
from bigram import *
from gpt import *

device = torch.device('mps')
torch.manual_seed(1337)

with open('./input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

# get all unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# tokenize characters into integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

model = GPT(vocab_size, device)
model.load_state_dict(torch.load("/Users/dinesh/code os/gpt/model.pt"))
model.to(device)

context = torch.ones((1, 1), dtype=torch.long, device=device)
model.generate(context, max_new_tokens=10000, decode=decode)