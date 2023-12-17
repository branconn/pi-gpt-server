import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams 
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = torch.device("mps") if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(42)

# load text
with open('training_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

# token mapping
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# data loading
def get_batch(split:str):
    dat = train_data if split == 'train' else val_data
    ix = torch.randint(len(dat) - block_size, (batch_size,))
    x = torch.stack([dat[i:i+block_size] for i in ix])
    y = torch.stack([dat[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

@torch.no_grad() # we do not intend to do backprop
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses =  torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss  = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attn"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        wei = q @ k.transpose(-2,-1) * C **-0.5 # (B,T,16) @ (B,16,T) ---> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out
    

class MultiHeadAttention(nn.Module):
    """multiple attentions in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    

class FeedForward(nn.Module):
    """simple linear layer followed by nonlinearity (per-token)"""

    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """transformer block: comm followed by comp"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # layer norms have beta and gamma trainiable params
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # "x +"" -> residual connections
        x = x + self.ffwd(self.ln2(x))
        return x


# simplementation of bigram
class BigramLanguageModel(nn.Module):
    '''
    bigram only looks at the previous character in predicting the next
    '''
    def __init__(self):
        super().__init__()
        # 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # number of embedding dimensions
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = Head(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        #
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T,)
            loss = F.cross_entropy(logits, targets)
        else: 
            loss = None

        return logits, loss
    
    def generate(self, idx, max_new_tok):
        # idx is (B,T) array if indicies in current context
        for _ in range(max_new_tok):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) # get predictions
            logits = logits[:, -1, :] # look at last timestep
            probs = F.softmax(logits, dim=-1) # get probabilities from softmax
            idx_next = torch.multinomial(probs, num_samples=1) # sample from prob dist
            idx = torch.cat((idx, idx_next), dim=1) # append sample
        return idx


model = BigramLanguageModel()
m = model.to(device)

print(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tok=500)[0].tolist()))
