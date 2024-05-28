import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
block_size = 256 # maximum context length for predictions
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

class Head(nn.Module):
    # single masked self-attention head
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x) # (B, T, C) -> (B, T, hs)
        q = self.query(x)   # (B, T, C) -> (B, T, hs)
        v = self.value(x)   # (B, T, C) -> (B, T, hs)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1])**(-0.5)    # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v   # (B, T, T) @ (B, T, hs) -> (B, T, hs) 
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    "feed forward network"

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-5):
        super().__init__()

        shape = (1, num_features)
        self.eps = eps

        # scale and shift parameters (trained with backprop)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, X):
        mean = X.mean(axis = 1, keepdim = True)
        var = X.var(axis = 1, keepdim = True)

        X_hat = (X - mean) / (torch.sqrt(var) + self.eps)

        return self.gamma * X_hat + self.beta

class Block(nn.Module):
    def __init__(self, num_embd, num_head):
        # n_embd: number of embeddings in embedding dimension, n_head: the number of heads 
        super().__init__()

        head_size = num_embd // num_head
        self.sa = MultiHeadAttention(num_head, head_size)
        self.ln1 = LayerNorm(n_embd)
        self.ffwd = FeedFoward(num_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, device):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.num_heads = 4

        self.word_token_embding = nn.Embedding(vocab_size, n_embd) # embedding table for word tokens  
        self.word_position_embding = nn.Embedding(block_size, n_embd) # embedding table for word positions 
        self.blocks = nn.Sequential(*[Block(n_embd, self.num_heads) for _ in range(n_layer)])
        self.ln = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) 

    # idx is the current time dimension
    def forward(self, idx, targets=None):
        B, T = idx.shape

        word_token_embd = self.word_token_embding(idx)    # (B, T, C) -> (B, T, NE)
        word_pos_embd = self.word_position_embding(torch.arange(T, device=self.device)) # (T, NE)
        x = word_token_embd + word_pos_embd # (B, T, NE)        
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)   # (B, T, NE) -> (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b*t, c)
            targets = targets.view(b*t)     
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens, decode):
        # idx is the (b, t) array of indices in the current context
        for _ in range(max_new_tokens):
            item = decode([idx[0][-1].item()])
            print(item, sep=' ', end='', flush=True)
            # get predictions
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # get the last time step from the predictions i.e (b, c)
            probs = F.softmax(logits, dim =-1) # softmax the logits (b, c)
            idx_next = torch.multinomial(probs, num_samples=1) # get the next index from the distribution # (b, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (b, t+1)
        return idx