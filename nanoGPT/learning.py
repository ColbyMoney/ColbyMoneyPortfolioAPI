import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

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

print(xb)
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        #idx and target are both (batch, time) tensor of integers
        logits = self.token_embedding_table(idx) # (batch, time, vocab_size)
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

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))