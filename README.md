# MiniGPT: Building a Decoder-Only Transformer from First Principles

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/Focus-Educational%20%2F%20First%20Principles-success.svg)

## Project Overview

**MiniGPT** is a minimalist, hands-on implementation of a decoder-only Transformer language model built in PyTorch. Created as a first-principles engineering exercise, this project strips away high-level framework abstractions to expose the core mathematical and architectural mechanics powering modern Large Language Models (LLMs)—including **causal self-attention**, **multi-head attention**, **token and positional embeddings**, **residual connections with Pre-LayerNorm**, and **autoregressive next-token prediction**.

> **Note on Tokenization Strategy:**  
> To keep the data pipeline transparent and easy to inspect during training, this implementation uses a **custom word-level tokenizer** (`text.split()` vocabulary mapping with an explicit `<END>` boundary marker) rather than a production subword algorithm like Byte-Pair Encoding (BPE) or WordPiece. While production LLMs operate on subword tokens to handle open vocabularies efficiently, the downstream tensor operations—embedding lookups, causal masking, attention score computation, cross-entropy optimization, and multinomial sampling—follow the exact foundational pipeline of full-scale generative models.

---

## Repository Structure

```text
.
├── minidemo.py             # Corpus preparation, word-level tokenizer, MiniGPT model, training loop, and inference
├── transformer_blocks.py   # Modular Transformer components: SelfAttentionHead, MultiHeadAttention, FeedForward, and Block
├── requirements.txt        # Python package dependencies
├── .gitignore              # Standard Python & PyTorch ignore rules
└── README.md               # Project documentation and architectural walkthrough
```

---

## Key Concepts Explored

1. **Word-Level Tokenization & Numericalization**: Building bidirectional lookup tables (`word_to_idx` and `idx_to_word`) and appending `<END>` stop markers so the model learns sequence boundaries.
2. **Sliding-Window Context Batching**: Constructing input sequences (`x`) and 1-step shifted target sequences (`y`) of length `block_size = 6` to train the model on every prefix simultaneously.
3. **Dual Embedding Spaces**: Combining semantic identity (`token_embedding`) with sequence order (`position_embedding`) in a `32`-dimensional vector space (`embedding_dim = 32`).
4. **Scaled Dot-Product Causal Self-Attention**: Computing Query ($Q$), Key ($K$), and Value ($V$) projections and applying a lower-triangular mask (`torch.tril`) so tokens only attend to past and present positions.
5. **Multi-Head Attention & Pointwise Feed-Forward Networks**: Splitting the embedding space across parallel attention heads (`n_heads = 2`), projecting concatenated outputs, and expanding representations through a $4\times$ hidden MLP (`FeedForward`).
6. **Autoregressive Generation**: Conditioning on a rolling context window (`idx[:, -block_size:]`), converting final-step logits into a probability distribution via `softmax`, and sampling the next token iteratively with `torch.multinomial`.

---

## Code Walkthrough

### 1. Causal Self-Attention (`transformer_blocks.py`)

At the heart of the decoder architecture is `SelfAttentionHead`. Each token emits a **Query** (what it is looking for), a **Key** (what it contains), and a **Value** (what information it passes forward).

```python
class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, block_size, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out
```

- **Scaled Dot-Product (`q @ k.transpose(-2, -1) / (C ** 0.5)`)**: Prevents large dot-product magnitudes from pushing the softmax function into regions with vanishingly small gradients.
- **Causal Masking (`self.tril`)**: Registered as a non-trainable buffer, the lower-triangular matrix sets future token positions to `-inf` prior to `F.softmax`, strictly preventing data leakage from future tokens during training.

---

### 2. Transformer Block with Pre-LayerNorm & Residual Pathways (`transformer_blocks.py`)

Each `Block` pairs communication across tokens (`MultiHeadAttention`) with per-token computation (`FeedForward`), wrapped in Pre-LayerNorm and residual skip connections (`x = x + ...`) to stabilize gradient flow:

```python
class Block(nn.Module):
    def __init__(self, embedding_dim, block_size, n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads)
        self.ffwd = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

---

### 3. The `MiniGPT` Model & Autoregressive Sampling (`minidemo.py`)

`MiniGPT` stacks `n_layers = 2` Transformer blocks, applies a final layer normalization (`self.ln_f`), and projects hidden states back to `vocab_size` logits (`self.head`):

```python
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
```

- **Context Cropping (`idx[:, -block_size:]`)**: Ensures the input sequence never exceeds the maximum length supported by `self.position_embedding`.
- **Cross-Entropy Training Objective**: Flattens `(B, T, C)` logits into `(B*T, C)` so PyTorch's `F.cross_entropy` evaluates next-token predictions across every batch item and timestep simultaneously.

---

## Hyperparameters at a Glance

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `block_size` | `6` | Maximum context length (in word tokens) for attention and positional embeddings |
| `embedding_dim` | `32` | Dimensionality of token and positional embedding vectors |
| `n_heads` | `2` | Number of parallel attention heads per block (`head_size = 16`) |
| `n_layers` | `2` | Number of stacked Transformer decoder blocks |
| `lr` | `1e-3` | Learning rate for the `AdamW` optimizer |
| `epochs` | `1500` | Number of training iterations with random mini-batches (`batch_size = 16`) |

---

## Setup & Usage

### 1. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training and Text Generation

Execute `minidemo.py` from the project directory to train the model for 1,500 steps and sample a 15-token continuation starting from the prompt `"hello"`:

```bash
python minidemo.py
```

**Example Output:**

```text
Using PyTorch version: 2.x.x
Using CUDA version: False
Using GPU: CPU
Step 0, Loss: 3.6124
Step 300, Loss: 0.3142
Step 600, Loss: 0.2418
Step 900, Loss: 0.2195
Step 1200, Loss: 0.2084
hello friend how are you doing today <END> The cat is sleeping on the couch <END>
```

---

## What's Next

Building `MiniGPT` with a simplified word-level vocabulary laid the groundwork for understanding tensor shapes, attention masks, and autoregressive loops. The upcoming milestones on this engineering roadmap focus on scaling toward a full, production-grade LLM from scratch:

- **Subword Tokenization (BPE / WordPiece)**: Replacing whitespace splitting with a custom Byte-Pair Encoding (BPE) and WordPiece tokenizer capable of handling arbitrary text, punctuation, and out-of-vocabulary words without exploding the embedding matrix.
- **Advanced Positional Encodings**: Moving beyond learned absolute position lookup tables (`nn.Embedding(block_size, embedding_dim)`) to **Rotary Position Embeddings (RoPE)** and relative positional encodings for better length generalization.
- **Optimized Multi-Head & Grouped-Query Attention**: Refactoring `MultiHeadAttention` from a `ModuleList` of separate linear layers into a single fused $QKV$ matrix multiplication with FlashAttention compatibility, dropout regularization, and KV-caching during inference.
- **Scaled Training Pipelines**: Introducing large-scale text datasets, validation splits, cosine learning rate schedules with warmup, gradient clipping, mixed-precision training (`bfloat16`/`fp16`), and model checkpointing.

