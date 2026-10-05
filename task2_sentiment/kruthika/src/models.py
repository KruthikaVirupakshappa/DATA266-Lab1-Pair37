"""Task 2.2 - three sentiment classifiers, all with word embeddings learned from scratch.

Each model maps (ids [B, T], lengths [B]) -> one logit per review; sigmoid(logit) = P(positive).
"""
import torch
import torch.nn as nn


def _mask(ids):
    return (ids != 0).unsqueeze(-1)  # (B, T, 1): True for real tokens, False for padding


class MeanEmbedding(nn.Module):
    """Baseline: average of word embeddings -> linear layer (fastText-style bag of words).
    Ignores word order entirely; fast and a strong reference point."""

    def __init__(self, vocab_size, embed_dim, dropout, **_):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, ids, lengths):
        m = _mask(ids)
        mean = (self.emb(ids) * m).sum(1) / m.sum(1).clamp(min=1)
        return self.fc(self.drop(mean)).squeeze(-1)


class BiLSTM(nn.Module):
    """Experimental 1: bidirectional LSTM reads the review left-to-right and right-to-left,
    so each position sees its context (e.g. "not ... good"). Max- and mean-pooling over time
    combine the strongest signals with the overall tone."""

    def __init__(self, vocab_size, embed_dim, hidden, layers, dropout, **_):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(4 * hidden, 1)  # [max-pool ; mean-pool] of 2*hidden each

    def forward(self, ids, lengths):
        x = self.drop(self.emb(ids))
        # Packing skips the padding, so the backward direction starts at the real last word.
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        m = _mask(ids)
        max_pool = out.masked_fill(~m, float("-inf")).max(1).values
        mean_pool = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return self.fc(self.drop(torch.cat([max_pool, mean_pool], dim=1))).squeeze(-1)


class TextCNN(nn.Module):
    """Experimental 2: 1-D convolutions with 3-, 4- and 5-word filters detect local phrases
    ("not worth it", "highly recommend"); max-over-time keeps each filter's strongest match."""

    def __init__(self, vocab_size, embed_dim, filters, kernel_sizes, dropout, **_):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, filters, k) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernel_sizes), 1)
        self.min_len = max(kernel_sizes)

    def forward(self, ids, lengths):
        if ids.size(1) < self.min_len:  # sequences must be at least as long as the widest filter
            ids = nn.functional.pad(ids, (0, self.min_len - ids.size(1)))
        x = self.emb(ids).transpose(1, 2)  # (B, embed_dim, T) for Conv1d
        pooled = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.fc(self.drop(torch.cat(pooled, dim=1))).squeeze(-1)


MODELS = {"mean_embedding": MeanEmbedding, "bilstm": BiLSTM, "textcnn": TextCNN}


def build_model(cfg, vocab_size):
    return MODELS[cfg["type"]](vocab_size=vocab_size, **cfg)
