import torch
from bigram import *
from gpt import *

device = torch.device('mps')
torch.manual_seed(1337)

# hyperparameters
batch_size = 64 # number of independent sequences to process in parallel
max_iters = 5000
eval_interval = 500
eval_iters = 200
lr = 3e-4

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

# train test splits
data = torch.tensor(encode(text), dtype = torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPT(vocab_size, device)
model.to(device)
print("Number of parameters: {} M".format(sum(p.numel() for p in model.parameters())/1e6))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "/Users/dinesh/code os/gpt/model.pt")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
model.generate(context, max_new_tokens=10000, decode=decode)