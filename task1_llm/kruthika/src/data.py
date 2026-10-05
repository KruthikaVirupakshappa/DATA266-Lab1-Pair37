"""Task 1.1 - character-level data preprocessing for TinyStories.

Pipeline: raw stories -> clean text -> train/val split -> char vocab -> integer ids
-> random fixed-length (x, y) windows where y is x shifted left by one character.
"""
import random

import torch

# Special character placed between stories so the model can learn where a story ends.
# Cleaning removes all non-ASCII characters, so this can never collide with real text.
EOS = "§"

# The raw data contains UTF-8 text that was wrongly decoded as Windows-1252
# ("mojibake"), e.g. a curly quote became "â€œ". Map these back to plain ASCII.
# Order matters: the bare "â€" (broken closing quote) must be replaced last.
MOJIBAKE = [
    ("â€œ", '"'), ("â€™", "'"), ("â€˜", "'"), ("â€¦", "..."),
    ("â€”", "-"), ("â€“", "-"), ("â€", '"'),
    ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("…", "..."), ("—", "-"), ("–", "-"),
    ("\xa0", " "), ("\t", " "),
]


def clean_text(text):
    """Repair mojibake, then keep only printable ASCII plus newlines."""
    for bad, good in MOJIBAKE:
        text = text.replace(bad, good)
    text = "".join(c for c in text if c == "\n" or 32 <= ord(c) < 127)
    return text.strip()


def _sample_clean(split, n, seed, exclude=frozenset()):
    """Randomly pick n cleaned, non-empty stories from a Hugging Face split,
    skipping any story whose text is in `exclude`."""
    order = list(range(len(split)))
    random.Random(seed).shuffle(order)  # fixed seed -> the same "random" sample every run
    picked, pos = [], 0
    while len(picked) < n:
        if pos >= len(order):
            raise ValueError(f"split has only {len(picked)} usable stories, need {n}")
        chunk = order[pos:pos + n - len(picked) + 1000]  # small buffer for skipped stories
        pos += len(chunk)
        for text in split.select(chunk)["text"]:
            text = clean_text(text or "")  # drop empty/malformed entries
            if text and text not in exclude and len(picked) < n:
                picked.append(text)
    return picked


def load_stories(path, n_train, n_val, seed):
    """Load the saved Hugging Face dataset folder and return (train_texts, val_texts).

    Validation comes from the official `validation` split and training from the
    `train` split, so the two never share a source. The train split also contains
    exact copies of some validation stories, so any training story identical to a
    chosen validation story is skipped - the 10K validation stories stay unseen.
    """
    from datasets import load_from_disk

    ds = load_from_disk(str(path))
    val = _sample_clean(ds["validation"], n_val, seed)
    train = _sample_clean(ds["train"], n_train, seed, exclude=set(val))
    return train, val


def build_vocab(texts):
    """Every distinct character in the training text, plus EOS, sorted for a stable order."""
    chars = sorted(set("".join(texts)) | {EOS})
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode(text, char_to_idx):
    """str -> list[int]. Characters not in the vocabulary are skipped."""
    return [char_to_idx[c] for c in text if c in char_to_idx]


def decode(ids, idx_to_char):
    """list[int] -> str."""
    return "".join(idx_to_char[i] for i in ids)


def encode_split(texts, char_to_idx):
    """Join all stories (each followed by EOS) into one long 1-D tensor of ids."""
    return torch.tensor(encode(EOS.join(texts) + EOS, char_to_idx), dtype=torch.long)


def get_batch(ids, block_size, batch_size, device, generator=None):
    """Sample `batch_size` random windows of `block_size` characters.

    x = ids[i : i+block_size], y = ids[i+1 : i+block_size+1], so y[t] is the
    character that comes right after x[t] - the next-character prediction target.
    """
    starts = torch.randint(len(ids) - block_size - 1, (batch_size, 1), generator=generator)
    offsets = (starts + torch.arange(block_size)).to(ids.device)  # (B, T) positions to read
    # One gather for the whole batch; if `ids` already lives on the GPU nothing is copied.
    return ids[offsets].to(device), ids[offsets + 1].to(device)
