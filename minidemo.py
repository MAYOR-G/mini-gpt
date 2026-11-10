import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformer_blocks import Block

print("Using PyTorch version:", torch.__version__)
print("Using CUDA version:", torch.cuda.is_available())
print("Using GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

corpus = [
    "hello friend how are you ",
    "hello friend how are you doing today ",
    "The tea is hot and delicious ",
    "The weather is nice and sunny ",
    "The cat is sleeping on the couch ",
    "The dog is barking at the mailman ",
    "The bird is singing in the tree ",
    "The fish is swimming in the pond ",
    "The rabbit is hopping in the garden ",
    "The horse is running in the field ",
    "i love programming in python ",
]

corpus = [s + " <END>" for s in corpus]
text = " ".join(corpus)
#print(text)

# a mini replacement for the tokenizer
words = list(set(text.split()))
vocab_size = len(words)
#print(words)
#print("Vocabulary size:", vocab_size)

word_to_idx = {word: i for i, word in enumerate(words)}
#print(word_to_idx)

idx_to_word = {i: word for i, word in enumerate(words)}
#print(idx_to_word)

data = torch.tensor([word_to_idx[word] for word in text.split()], dtype=torch.long)
#print(data)

block_size = 6
embedding_dim = 32
n_heads = 2
n_layers = 2
lr = 1e-3
epochs = 1500

def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y  

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, n_heads, n_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding(torch.arange(T, device=idx.device))
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx 
    
model = MiniGPT(vocab_size, embedding_dim, block_size, n_heads, n_layers)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(epochs):
    x, y = get_batch()
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        print(f"Step {step}, Loss: {loss.item()}")
        
# Generate text (test the model)
context = torch.tensor([[word_to_idx["hello"]]], dtype=torch.long)
generated_idx = model.generate(context, max_new_tokens=15)

print(" ".join([idx_to_word[idx.item()] for idx in generated_idx[0]]))
