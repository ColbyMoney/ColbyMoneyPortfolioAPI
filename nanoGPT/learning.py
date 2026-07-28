import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path(__file__).parent

with open(DATA_DIR / "shakespeare.txt", 'r', encoding='utf-8') as f:
    shakespeare_text = f.read()
print(len(shakespeare_text), "characters loaded from shakespeare.txt")

chars = sorted(list(set(shakespeare_text)))
vocab_size = len(chars)
print("Possible token characters: {", ''.join(chars), "}")
print("Vocabulary size: ", vocab_size)

chars_to_tokens = { ch: i for i, ch in enumerate(chars)}
tokens_to_chars = { i: ch for i, ch in enumerate(chars)}
def encode(char_sequence):
    return [chars_to_tokens[c] for c in char_sequence]
def decode(token_sequence):
    return ''.join([tokens_to_chars[i] for i in token_sequence])
print("Encoding and decoding example: 'Hello world!'")
print(encode("Hello world!"))
print(decode(encode("Hello world!")))

shakespeare_tokens = torch.tensor(encode(shakespeare_text), dtype=torch.long)
print("Shakespeare tokens shape: ", shakespeare_tokens.shape, " Shakespeare tokens dtype: ", shakespeare_tokens.dtype)
print("Shakespeare first 100 tokens: ", shakespeare_tokens[:100])

training_validation_divider = int(0.9 * len(shakespeare_tokens))
shakespeare_training_tokens = shakespeare_tokens[:training_validation_divider] # first 90% of tokens for training
shakespeare_validation_tokens = shakespeare_tokens[training_validation_divider:] # last 10% of tokens for validation

context_size = 8
shakespeare_training_tokens[:context_size + 1]
x = shakespeare_training_tokens[:context_size]
y = shakespeare_training_tokens[1:context_size+1]
for t in range(context_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context}, the target is: {target}")

torch.manual_seed(1337)
batch_size = 4
context_size = 8
def get_batch(split):
    data = shakespeare_training_tokens if split == 'train' else shakespeare_validation_tokens
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    return x, y
xb, yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)
print("-----")
for batch in range(batch_size):
    for time in range(context_size):
        context = xb[batch, :time+1]
        target = yb[batch, time]
        print(f"When input is {context}, the target is: {target}")

print(xb) # our input to the transformer
torch.manual_seed(1337)
n_embd = 32
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        #idx and target are both (batch, time) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (batch, time, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (time, n_embd)
        x = tok_emb + pos_emb # (batch, time, n_embd)
        logits = self.lm_head(x) # (batch, time, vocab_size)
        if targets is None:
            loss = None
        else:
            batch, time, channels = logits.shape
            logits = logits.view(batch * time, channels)
            targets = targets.view(batch * time)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        # idx is (batch, time) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (batch, channels)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (batch, channels)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (batch, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (batch, time+1)
        return idx
m = BigramLanguageModel()
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
context_size = 32
for steps in range(10000):
    # sample a batch of data
    xb, yb = get_batch("train")
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))



# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)

torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
x.shape

# version 1: We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)

# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
print(wei);
xbow2 = wei @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)
print(torch.allclose(xbow, xbow2))
print(xbow[0])
print(xbow2[0])

# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
print(xbow3[0])
print(torch.allclose(xbow, xbow3))