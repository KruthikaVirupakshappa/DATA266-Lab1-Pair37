"""Task 1 metric helpers (formula-level only; model-agnostic).

Agree on these definitions with your teammate so numbers are comparable across members.
"""
import math

import torch


def perplexity(ce_loss):
    """exp(mean cross-entropy in nats)."""
    return math.exp(ce_loss)


def bits_per_char(ce_loss):
    """Cross-entropy converted from nats to bits (one token = one character here)."""
    return ce_loss / math.log(2)


def top1_accuracy(logits, targets):
    """Fraction of positions where argmax prediction == next character. logits (B,T,V), targets (B,T)."""
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def count_parameters(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def _units(text, unit):
    return text.split() if unit == "word" else list(text)


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts, n, unit="word"):
    """Unique n-grams / total n-grams, pooled over all generated samples (higher = more diverse)."""
    grams = [g for t in texts for g in _ngrams(_units(t, unit), n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repeated_ngram_rate(texts, n=4, unit="word"):
    """Share of n-grams that already appeared earlier in the same sample, averaged over samples
    (higher = more looping/repetition)."""
    rates = []
    for t in texts:
        grams = _ngrams(_units(t, unit), n)
        if grams:
            rates.append(1 - len(set(grams)) / len(grams))
    return sum(rates) / len(rates) if rates else 0.0


def grad_norm(model):
    """Global L2 norm of all parameter gradients (call after backward, before optimizer.step)."""
    norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
    return torch.norm(torch.stack(norms), 2).item() if norms else 0.0
