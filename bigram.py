import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams 
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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

# simplementation of bigram
class BigramLanguageModel(nn.Module):
    '''
    bigram only looks at the previous character in predicting the next
    '''
    def __init__(self, vocab_size):
        super().__init__()
        # 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        #
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T,)
            # cross_entropy expects channels as second dim
            loss = F.cross_entropy(logits, targets)
        else: 
            loss = None

        return logits, loss
    
    def generate(self, idx, max_new_tok):
        # idx is (B,T) array if indicies in current context
        for _ in range(max_new_tok):
            logits, loss = self(idx) # get predictions
            logits = logits[:, -1, :] # look at last timestep
            probs = F.softmax(logits, dim=-1) # get probabilities from softmax
            idx_next = torch.multinomial(probs, num_samples=1) # sample from prob dist
            idx = torch.cat((idx, idx_next), dim=1) # append sample
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

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
