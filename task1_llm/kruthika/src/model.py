"""Task 1.2 - GPT-style character language model, built from scratch.

Only basic layers are used (nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout);
attention, masking and the transformer block are written by hand.

    ids (B,T) -> token emb + position emb -> [Block] x n_layer -> LayerNorm -> LM head -> logits (B,T,V)
    Block:  x = x + Attention(LayerNorm(x))
            x = x + FeedForward(LayerNorm(x))
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position can only look at itself and earlier positions."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        # One linear layer produces queries, keys and values for all heads at once.
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)  # mixes the heads' outputs back together
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # Lower-triangular matrix: mask[i, j] = 1 if j <= i (position i may attend to j).
        # A buffer moves with .to(device) and is saved in the state_dict, but is not trained.
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): each head works on its own slice of C.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: how well each query matches each key.
        # Dividing by sqrt(head_dim) keeps scores from growing with dimension,
        # which would make softmax saturate and gradients vanish.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, h, T, T)
        # Causal mask: future positions get -inf, so softmax gives them weight 0.
        scores = scores.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        out = weights @ v  # weighted average of values: (B, h, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # concatenate heads
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP: expand to 4x width, GELU non-linearity, project back."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block (pre-LayerNorm): attention = tokens talk to each other,
    feed-forward = each token processes what it gathered. Residual connections
    (x + ...) give gradients a direct path through deep stacks."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        """cfg keys: vocab_size, block_size, n_layer, n_head, n_embd, dropout."""
        super().__init__()
        self.block_size = cfg["block_size"]
        n_embd = cfg["n_embd"]
        dropout = cfg.get("dropout", 0.0)

        self.tok_emb = nn.Embedding(cfg["vocab_size"], n_embd)  # what each character is
        self.pos_emb = nn.Embedding(self.block_size, n_embd)    # where it sits (learned)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, cfg["n_head"], self.block_size, dropout) for _ in range(cfg["n_layer"])]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, cfg["vocab_size"])  # hidden state -> score per character

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Small random weights (GPT-2 style) keep the initial predictions close to
        # uniform, so the starting loss is about ln(vocab_size).
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy over every position: flatten (B, T) into B*T predictions.
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Append one sampled character at a time.

        temperature=0 -> greedy (always the most likely char); <1 sharper, >1 more random.
        top_k -> only sample from the k most likely characters.
        """
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size:]  # model only sees the last block_size chars
            logits, _ = self(context)
            logits = logits[:, -1, :]  # prediction for the next character only
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
