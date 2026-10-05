"""Build the shared Task 1 test set used to compare both members' models.

5,000 stories from the official TinyStories `validation` split (revision f54c09f...) that
neither model has seen:
  - not in Kruthika's 10K validation set (her training stories come from the train split), and
  - not anywhere in the `train` split - Jiayi's training AND validation stories are slices of
    the train split, and Kruthika's training stories too. A story is excluded if its raw text
    (Jiayi's preprocessing) or its cleaned text (Kruthika's preprocessing) occurs in `train`.
Duplicates inside the validation split are removed as well.

Usage (from the repo root; needs task1_llm/data/tinystories from download_tinystories.py):
    python task1_llm/shared_eval/build_test_set.py
Writes task1_llm/shared_eval/test_set.json (validation-split indices + SHA-1 of each raw story).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "task1_llm" / "kruthika" / "src"))
import data as kdata  # noqa: E402  (Kruthika's preprocessing: clean_text, load_stories)

N_TEST = 5000
SEED = 266
OUT = Path(__file__).resolve().parent / "test_set.json"


def h(text):
    return hashlib.sha1(text.encode("utf-8")).digest()


def main():
    ds = load_from_disk(str(ROOT / "task1_llm" / "data" / "tinystories"))
    val_raw = ds["validation"]["text"]

    print("hashing the train split (raw and cleaned)...")
    train_raw, train_clean = set(), set()
    for t in ds["train"]["text"]:
        t = t or ""
        train_raw.add(h(t.strip()))
        train_clean.add(h(kdata.clean_text(t)))

    print("rebuilding Kruthika's 10K validation set (seed 1337)...")
    _, k_val = kdata.load_stories(ROOT / "task1_llm" / "data" / "tinystories", 100000, 10000, 1337)
    k_val = {h(t) for t in k_val}

    stats = {"validation_split": len(val_raw)}
    seen, candidates = set(), []
    for i, t in enumerate(val_raw):
        t = t or ""
        c = kdata.clean_text(t)
        reason = None
        if not t.strip() or not c:
            reason = "empty"
        elif h(c) in k_val:
            reason = "in_kruthika_val"
        elif h(t.strip()) in train_raw or h(c) in train_clean:
            reason = "in_train_split"
        elif h(t.strip()) in seen:
            reason = "duplicate_in_validation"
        if reason:
            stats[reason] = stats.get(reason, 0) + 1
            continue
        seen.add(h(t.strip()))
        candidates.append(i)
    stats["eligible"] = len(candidates)
    print(stats)

    rng = np.random.default_rng(SEED)
    chosen = sorted(int(i) for i in rng.choice(candidates, N_TEST, replace=False))
    OUT.write_text(json.dumps({
        "dataset": "roneneldan/TinyStories", "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "split": "validation", "seed": SEED, "n": len(chosen), "selection_stats": stats,
        "indices": chosen,
        "sha1_raw": [hashlib.sha1(val_raw[i].encode("utf-8")).hexdigest() for i in chosen],
    }, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(chosen)} stories, "
          f"{sum(len(val_raw[i]) for i in chosen):,} raw characters")


if __name__ == "__main__":
    main()
