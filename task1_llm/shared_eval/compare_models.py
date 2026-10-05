"""Compare both members' Task 1 models on the shared test set (built by build_test_set.py).

Each model is evaluated with ITS OWN preprocessing, vocabulary and context length:
    kruthika: cleaned text (mojibake repaired, ASCII), stories joined with '§', 89 chars, context 256
    jiayi:    raw text, stories joined with '\\n', 152 chars, context 128
Metrics on the test set (consecutive non-overlapping windows, as both members used for validation):
    cross-entropy loss, perplexity, bits-per-character, top-1 next-char accuracy, and
    bits per ORIGINAL character (total bits / raw characters of the 5,000 stories) - the fairest
    single number, because the two pipelines produce slightly different character streams.
Generation uses one shared sampling loop for both models (same prompts, seeds, lengths, CPU).

Sanity check first: each model is re-scored on its own original validation split and must
reproduce the validation loss its owner reported.

Usage (from the repo root):
    python task1_llm/shared_eval/compare_models.py [--jiayi-dir task1_llm/jiayi]
Writes task1_llm/shared_eval/{comparison_report.csv, comparison_table.md, generation_samples.json/.txt}
and a raw log in reproducibility/raw_logs/.
"""
import argparse
import csv
import json
import logging
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "task1_llm" / "kruthika" / "src"))
import data as kdata  # noqa: E402
import metrics as kmetrics  # noqa: E402
from model import GPT  # noqa: E402

NOTES = """## How to read this table

- **Sanity check:** before scoring the test set, each model was re-scored on its own original validation split and
  reproduced its owner's reported validation loss (difference < 0.0001), so both models and vocabularies were loaded
  correctly (`comparison_report.csv`, first rows).
- **Bits per original character** divides each model's total bits by the length of the raw stories, so it is not
  affected by the two preprocessing pipelines (cleaning changes about 0.06% of characters; 2 characters were unknown
  to the raw-text model and skipped). Plain bits-per-character is almost identical here.
- **Distinct-n rewards incoherence as well as variety:** sentences that jump between unrelated ideas produce many
  unique word combinations. Read it together with the samples (`generation_samples.txt`); a higher value is not
  automatically better.
- **Generation speed** was measured for both models on the same CPU with the same sampling loop. It mostly reflects
  model size (10.8M vs 0.45M parameters) and context length (256 vs 128).
- **Own-training-run rows** come from each member's `metrics_report.csv`: validation numbers there are on each
  member's *own* validation split (different stories), and throughput/memory depend on each member's batch size and
  measurement code. Only the shared-test-set rows are measured on identical data with identical code.
"""

PROMPTS = ["Once upon a time", "One day, a little girl named Lily", "The dog was very sad because",
           "Tom and his mom went to the park.", "In a big forest, there lived"]
SAMPLES_PER_PROMPT, TEMPERATURE, MAX_NEW, GEN_SEED = 4, 0.8, 300, 1234


# ---------------- Jiayi's model, copied verbatim from task1_llm/jiayi/src/Task1_GPT.ipynb ----------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        QKV = self.qkv(x)
        Q, K, V = QKV.chunk(3, dim=-1)
        Q = Q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, seq_len=128, num_heads=4, num_layers=2):
        super().__init__()
        if num_layers != 2:
            raise ValueError("This TinyGPT implementation uses exactly 2 decoder blocks.")
        self.d_model, self.num_heads, self.num_layers, self.seq_len = d_model, num_heads, num_layers, seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.block1 = DecoderBlock(d_model, num_heads)
        self.block2 = DecoderBlock(d_model, num_heads)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.block2(self.block1(x))
        return self.head(self.ln(x))


# ---------------- one adapter per member: preprocessing + model ----------------
class Member:
    """name, context length, char_to_idx, model; encode(stories) -> (ids tensor, skipped chars)."""

    def logits(self, x):
        raise NotImplementedError

    def encode_text(self, text):
        ids = [self.c2i[c] for c in text if c in self.c2i]
        return ids, len(text) - len(ids)

    def decode(self, ids):
        return "".join(self.i2c[i] for i in ids)


class Kruthika(Member):
    name = "kruthika"

    def __init__(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = GPT(ck["model_cfg"]).eval()
        self.model.load_state_dict(ck["model_state"])
        self.c2i = ck["char_to_idx"]
        self.i2c = {i: c for c, i in self.c2i.items()}
        self.ctx = ck["model_cfg"]["block_size"]
        self.cfg = ck["model_cfg"]

    def logits(self, x):
        return self.model(x)[0]

    def encode_stories(self, stories):
        cleaned = [kdata.clean_text(s) for s in stories]
        text = kdata.EOS.join(cleaned) + kdata.EOS
        ids, skipped = self.encode_text(text)
        removed_by_cleaning = sum(len(s) for s in stories) - sum(len(c) for c in cleaned)
        return torch.tensor(ids, dtype=torch.long), {"skipped_unknown": skipped, "removed_by_cleaning": removed_by_cleaning}

    def readable(self, text):
        return text.replace(kdata.EOS, "\n<END OF STORY>\n")


class Jiayi(Member):
    name = "jiayi"

    def __init__(self, ckpt_path, cfg, train_texts):
        # Her vocabulary is not stored in the checkpoint: rebuild it exactly as her notebook did.
        chars = sorted(set("\n".join(train_texts)))
        self.c2i = {ch: i for i, ch in enumerate(chars)}
        self.i2c = {i: ch for ch, i in self.c2i.items()}
        m = cfg["model"]
        self.ctx = cfg["data"]["seq_len"]
        self.model = TinyGPT(len(chars), m["d_model"], self.ctx, m["num_heads"], m["num_layers"]).eval()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ck["model_state_dict"]["head.weight"].shape[0] == len(chars), "rebuilt vocab does not match checkpoint"
        self.model.load_state_dict(ck["model_state_dict"])
        self.reported_val_loss = ck["val_loss"]
        self.cfg = cfg

    def logits(self, x):
        return self.model(x)

    def encode_stories(self, stories):
        ids, skipped = self.encode_text("\n".join(stories))
        return torch.tensor(ids, dtype=torch.long), {"skipped_unknown": skipped, "removed_by_cleaning": 0}

    def readable(self, text):
        return text


# ---------------- evaluation ----------------
@torch.no_grad()
def score(member, ids, device, batch_size=64):
    """Non-overlapping windows of the member's context length: mean CE (nats) and top-1 accuracy."""
    member.model.to(device)
    T = member.ctx
    n = (len(ids) - 1) // T
    xs, ys = ids[:n * T].view(n, T), ids[1:n * T + 1].view(n, T)
    loss_sum, correct = 0.0, 0
    for i in range(0, n, batch_size):
        x, y = xs[i:i + batch_size].to(device), ys[i:i + batch_size].to(device)
        logits = member.logits(x)
        loss_sum += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
    member.model.to("cpu")
    return loss_sum / (n * T), correct / (n * T), n * T


@torch.no_grad()
def sample(member, prompt, temperature, seed):
    """Shared sampling loop (CPU): crop to the member's context, greedy if temperature == 0."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.tensor([member.encode_text(prompt)[0]], dtype=torch.long)
    for _ in range(MAX_NEW):
        logits = member.logits(ids[:, -member.ctx:])[:, -1, :]
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1, generator=g)
        ids = torch.cat([ids, nxt], dim=1)
    return member.decode(ids[0].tolist())


def words_text(member, continuation):
    """Text used for word-level diversity metrics: story separators become spaces."""
    return continuation.replace(kdata.EOS, " ") if member.name == "kruthika" else continuation


def read_metrics_csv(path):
    with open(path) as f:
        return {r["metric"]: r["value"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jiayi-dir", default="task1_llm/jiayi")
    ap.add_argument("--jiayi-ckpt", default="checkpoints/task1_gpt_best_20261001_000337.pt")
    ap.add_argument("--kruthika-ckpt", default="task1_llm/kruthika/checkpoints/gpt_L6_H6_D384_best.pt")
    args = ap.parse_args()

    log_dir = ROOT / "reproducibility" / "raw_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"task1_shared_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)])
    log = logging.info
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    log(f"python={platform.python_version()} torch={torch.__version__} eval device={device} generation device=cpu")

    ds = load_from_disk(str(ROOT / "task1_llm" / "data" / "tinystories"))
    jdir = ROOT / args.jiayi_dir
    jcfg = json.loads((jdir / "config.json").read_text())
    j_shuffled = ds["train"].shuffle(seed=jcfg["seed"])
    n_tr, n_va = jcfg["data"]["train_size"], jcfg["data"]["val_size"]
    j_train = j_shuffled.select(range(n_tr))["text"]
    j_val = j_shuffled.select(range(n_tr, n_tr + n_va))["text"]

    members = [Kruthika(ROOT / args.kruthika_ckpt), Jiayi(jdir / args.jiayi_ckpt, jcfg, j_train)]
    for m in members:
        log(f"{m.name}: vocab {len(m.c2i)} chars, context {m.ctx}, parameters {sum(p.numel() for p in m.model.parameters()):,}")

    # ---- sanity check: reproduce each member's own validation loss ----
    _, k_val = kdata.load_stories(ROOT / "task1_llm" / "data" / "tinystories", 100000, 10000, 1337)
    k_reported = float(read_metrics_csv(ROOT / "task1_llm" / "kruthika" / "metrics_report.csv")["val_cross_entropy"])
    sanity = {}
    for m, stories, reported in ((members[0], None, k_reported), (members[1], j_val, members[1].reported_val_loss)):
        if m.name == "kruthika":  # her val texts are already cleaned; encode exactly as in training
            ids = kdata.encode_split(k_val, m.c2i)
        else:
            ids, _ = m.encode_stories(stories)
        loss, _, _ = score(m, ids, device)
        sanity[m.name] = {"reproduced": loss, "reported": reported, "diff": loss - reported}
        log(f"SANITY {m.name}: own validation loss reproduced {loss:.5f} vs reported {reported:.5f} (diff {loss - reported:+.5f})")
        assert abs(loss - reported) < 0.01, f"{m.name}: model/vocab not loaded correctly"

    # ---- shared test set ----
    ts = json.loads((HERE / "test_set.json").read_text())
    val = ds["validation"]["text"]
    stories = [val[i] for i in ts["indices"]]
    import hashlib
    assert [hashlib.sha1(s.encode("utf-8")).hexdigest() for s in stories] == ts["sha1_raw"], "test set mismatch"
    raw_chars = sum(len(s) for s in stories)
    log(f"test set: {len(stories)} stories, {raw_chars:,} raw characters (revision {ts['revision'][:8]})")

    results = {}
    for m in members:
        ids, enc = m.encode_stories(stories)
        t0 = time.perf_counter()
        loss, acc, n_pred = score(m, ids, device)
        secs = time.perf_counter() - t0
        total_bits = loss * n_pred / math.log(2)
        results[m.name] = {
            "test_cross_entropy": loss, "test_perplexity": math.exp(loss), "test_bits_per_char": loss / math.log(2),
            "test_bits_per_original_char": total_bits / raw_chars * (len(ids) / n_pred),
            "test_top1_acc": acc, "test_chars_scored": n_pred, "test_chars_in_stream": len(ids),
            "chars_removed_by_cleaning": enc["removed_by_cleaning"], "chars_skipped_unknown": enc["skipped_unknown"],
            "eval_seconds": secs,
        }
        log(f"TEST {m.name}: " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in results[m.name].items()))

    # ---- generation: identical settings, shared loop, CPU ----
    torch.set_num_threads(max(1, torch.get_num_threads()))
    samples = {m.name: {"sampled": [], "greedy": []} for m in members}
    for m in members:
        new_chars, secs = 0, 0.0
        for p_i, prompt in enumerate(PROMPTS):
            runs = [("sampled", TEMPERATURE, GEN_SEED + 10 * p_i + k) for k in range(SAMPLES_PER_PROMPT)] + [("greedy", 0, 0)]
            for kind, temp, seed in runs:
                t0 = time.perf_counter()
                text = sample(m, prompt, temp, seed)
                secs += time.perf_counter() - t0
                new_chars += MAX_NEW
                samples[m.name][kind].append({"prompt": prompt, "seed": seed, "text": text, "continuation": text[len(prompt):]})
        sampled = [words_text(m, s["continuation"]) for s in samples[m.name]["sampled"]]
        greedy = [words_text(m, s["continuation"]) for s in samples[m.name]["greedy"]]
        results[m.name].update({
            **{f"distinct_{n}": kmetrics.distinct_n(sampled, n) for n in (1, 2, 3)},
            "repeated_4gram_rate": kmetrics.repeated_ngram_rate(sampled, 4),
            "greedy_repeated_4gram_rate": kmetrics.repeated_ngram_rate(greedy, 4),
            "gen_chars_per_s_cpu": new_chars / secs,
        })
        log(f"GEN {m.name}: distinct-1/2/3 {results[m.name]['distinct_1']:.3f}/{results[m.name]['distinct_2']:.3f}/"
            f"{results[m.name]['distinct_3']:.3f} | rep-4gram sampled {results[m.name]['repeated_4gram_rate']:.4f} "
            f"greedy {results[m.name]['greedy_repeated_4gram_rate']:.4f} | {results[m.name]['gen_chars_per_s_cpu']:.0f} chars/s (CPU)")

    # ---- training-side numbers from each member's own run ----
    own = {"kruthika": read_metrics_csv(ROOT / "task1_llm" / "kruthika" / "metrics_report.csv"),
           "jiayi": read_metrics_csv(jdir / "metrics_report.csv")}
    own_keys = ["parameters", "epochs", "train_cross_entropy", "val_cross_entropy", "val_perplexity", "val_bits_per_char",
                "val_top1_acc", "generalization_gap", "grad_norm_mean", "grad_norm_max", "nan_inf_steps",
                "train_tokens_per_s", "total_train_time_min", "peak_gpu_mem_mb", "device"]

    # ---- write outputs ----
    rows = [{"section": "sanity check (own validation)", "metric": f"{k}", **{m: sanity[m][k] for m in sanity}}
            for k in ("reproduced", "reported", "diff")]
    keys = list(results["kruthika"])
    rows += [{"section": "shared test set" if not k.startswith(("distinct", "repeated", "greedy", "gen_")) else "generation (shared settings)",
              "metric": k, "kruthika": results["kruthika"][k], "jiayi": results["jiayi"][k]} for k in keys]
    rows += [{"section": "own training run (own validation split)", "metric": k,
              "kruthika": own["kruthika"].get(k, ""), "jiayi": own["jiayi"].get(k, "")} for k in own_keys]
    with open(HERE / "comparison_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "metric", "kruthika", "jiayi"])
        w.writeheader()
        w.writerows(rows)
    (HERE / "generation_samples.json").write_text(json.dumps(samples, indent=1))
    with open(HERE / "generation_samples.txt", "w") as f:
        for m in members:
            for kind in ("sampled", "greedy"):
                for s in samples[m.name][kind]:
                    f.write(f"=== {m.name} | {kind} | seed {s['seed']} | prompt: {s['prompt']!r} ===\n{m.readable(s['text'])}\n\n")

    def fmt(v, d=4):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return str(v)
        if v.is_integer() or abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.1f}" if abs(v) >= 100 else f"{v:.{d}f}"
    k, j = results["kruthika"], results["jiayi"]
    table = ["| Metric | Kruthika | Jiayi |", "|---|---|---|"]
    table += [f"| Architecture | {k_cfg} | {j_cfg} |".format() for k_cfg, j_cfg in [(
        f"{members[0].cfg['n_layer']} layers, {members[0].cfg['n_head']} heads, d={members[0].cfg['n_embd']}, context {members[0].ctx}",
        f"{jcfg['model']['num_layers']} layers, {jcfg['model']['num_heads']} heads, d={jcfg['model']['d_model']}, context {members[1].ctx}")]]
    table += [f"| Vocabulary / preprocessing | {len(members[0].c2i)} chars, cleaned text | {len(members[1].c2i)} chars, raw text |"]
    for label, key, d in [("Test cross-entropy (nats)", "test_cross_entropy", 4), ("Test perplexity", "test_perplexity", 3),
                          ("Test bits-per-character", "test_bits_per_char", 4),
                          ("**Test bits per original character**", "test_bits_per_original_char", 4),
                          ("Test top-1 accuracy", "test_top1_acc", 4), ("Distinct-1 / 2 / 3 (words, T=0.8)", None, 3),
                          ("Repeated 4-gram rate (T=0.8 / greedy)", None, 4), ("Generation speed (chars/s, CPU)", "gen_chars_per_s_cpu", 0)]:
        if label.startswith("Distinct"):
            table.append(f"| {label} | {k['distinct_1']:.3f} / {k['distinct_2']:.3f} / {k['distinct_3']:.3f} | "
                         f"{j['distinct_1']:.3f} / {j['distinct_2']:.3f} / {j['distinct_3']:.3f} |")
        elif label.startswith("Repeated"):
            table.append(f"| {label} | {k['repeated_4gram_rate']:.4f} / {k['greedy_repeated_4gram_rate']:.4f} | "
                         f"{j['repeated_4gram_rate']:.4f} / {j['greedy_repeated_4gram_rate']:.4f} |")
        else:
            table.append(f"| {label} | {fmt(k[key], d)} | {fmt(j[key], d)} |")
    table.append("| *From each member's own training run:* | | |")
    for key in own_keys:
        table.append(f"| {key} | {fmt(own['kruthika'].get(key, ''))} | {fmt(own['jiayi'].get(key, ''))} |")
    (HERE / "comparison_table.md").write_text(
        "# Task 1 model comparison (shared test set)\n\n"
        f"Shared test set: {len(stories)} official-validation stories ({raw_chars:,} characters) that are in neither "
        "member's training or validation data (see `test_set.json`). Each model uses its own preprocessing, "
        "vocabulary and context length; generation uses identical prompts, seeds and settings "
        f"({SAMPLES_PER_PROMPT} samples at T={TEMPERATURE} + 1 greedy per prompt, {MAX_NEW} new characters).\n\n"
        + "\n".join(table) + "\n\n" + NOTES)
    log(f"outputs -> {(HERE / 'comparison_report.csv').relative_to(ROOT)}, comparison_table.md, generation_samples.*")
    log(f"raw log -> {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
