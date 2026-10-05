"""Task 2 - Yelp Polarity sentiment classification: data analysis, preprocessing,
training of 3 models, full evaluation, comparison and error-candidate export.

Usage (from the repo root):
    python task2_sentiment/kruthika/src/train.py               # settings from src/config.yaml
    python task2_sentiment/kruthika/src/train.py --epochs 3    # change max epochs for all models
    python task2_sentiment/kruthika/src/train.py --force       # retrain models that already finished

A model whose checkpoint and test predictions already exist is not retrained, so after a
disconnect simply run the script again.

Writes (paths relative to task2_sentiment/kruthika/):
    data_processed/          vocab, preprocessing stats (+ cached encoded arrays, git-ignored)
    checkpoints/<model>_best.pt
    outputs/eda/             class balance and review-length plots, data-quality stats
    outputs/<model>/         training history, confusion matrix, test probabilities
    outputs/                 comparison table, slice metrics, curves, McNemar, error candidates
    metrics_report.csv       every Task 2 metric for every model, in one file
plus a raw log and a manifest in reproducibility/.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(SRC.parents[1] / "scripts"))  # task2_sentiment/scripts, for the downloader

import data  # noqa: E402
import evaluate as ev  # noqa: E402
import utils  # noqa: E402
from download_yelp_polarity import download_yelp_polarity  # noqa: E402
from models import build_model  # noqa: E402

MEMBER_DIR = utils.REPO_ROOT / "task2_sentiment" / utils.MEMBER


# ---------------- data ----------------
def eda(train_texts, train_labels, test_texts, test_labels, out_dir, log):
    """Brief 2.1.1-2.1.2: class balance, review lengths, missing/malformed entries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = {"quality": data.data_quality(train_texts, test_texts), "splits": {}}
    lengths = {}
    for split, texts, labels in (("train", train_texts, train_labels), ("test", test_texts, test_labels)):
        n_words = np.array([len(t.split()) for t in texts])
        lengths[split] = (n_words, labels)
        s = {"n": len(texts), "class_counts": {"negative": int((labels == 0).sum()), "positive": int((labels == 1).sum())}}
        for name, m in (("all", slice(None)), ("negative", labels == 0), ("positive", labels == 1)):
            w = n_words[m]
            s[f"words_{name}"] = {"mean": float(w.mean()), "median": float(np.median(w)), "p95": float(np.percentile(w, 95)),
                                  "min": int(w.min()), "max": int(w.max())}
        stats["splits"][split] = s
    log.info(f"data quality: {stats['quality']}")
    for split, s in stats["splits"].items():
        log.info(f"{split}: n={s['n']:,} classes={s['class_counts']} words mean={s['words_all']['mean']:.0f} "
                 f"median={s['words_all']['median']:.0f} p95={s['words_all']['p95']:.0f} max={s['words_all']['max']}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, split in enumerate(("train", "test")):
        cc = stats["splits"][split]["class_counts"]
        axes[0].bar([i - 0.2, i + 0.2], [cc["negative"], cc["positive"]], width=0.4,
                    color=["#d62728", "#2ca02c"], label=["negative", "positive"] if i == 0 else None)
    axes[0].set(xticks=[0, 1], xticklabels=["train", "test"], ylabel="reviews", title="Class distribution")
    axes[0].legend()
    n_words, labels = lengths["train"]
    for c, name, col in ((0, "negative", "#d62728"), (1, "positive", "#2ca02c")):
        axes[1].hist(n_words[labels == c], bins=100, range=(0, 600), alpha=0.5, label=name, color=col)
    axes[1].set(xlabel="words per review", ylabel="reviews", title="Review length (train)")
    axes[1].legend()
    axes[2].boxplot([n_words[labels == 0], n_words[labels == 1]], showfliers=False)
    axes[2].set(xticks=[1, 2], xticklabels=["negative", "positive"], ylabel="words per review",
                title="Review length by class (train, no outliers)")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "class_and_length.png", dpi=150)
    plt.close(fig)
    (out_dir / "eda_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def prepare(cfg, train_texts, train_labels, test_texts, log):
    """Preprocess, split, build vocab, encode. Cached by data-config hash so reruns are fast."""
    dc = cfg["data"]
    proc = MEMBER_DIR / "data_processed"
    proc.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(json.dumps([dc, cfg["seed"]], sort_keys=True).encode()).hexdigest()[:10]
    cache = proc / f"encoded_{key}.npz"
    if cache.exists():
        z = np.load(cache)
        vocab = json.loads((proc / "vocab.json").read_text())
        log.info(f"loaded cached preprocessing {utils.rel(cache)}")
        return {k: z[k] for k in z.files}, vocab

    t0 = time.perf_counter()
    stopwords = data.load_stopwords()
    tok_train = [data.preprocess(t, stopwords, dc["stemming"]) for t in train_texts]
    tok_test = [data.preprocess(t, stopwords, dc["stemming"]) for t in test_texts]
    tr_idx, va_idx = data.stratified_split(train_labels, dc["val_fraction"], cfg["seed"])
    vocab = data.build_vocab([tok_train[i] for i in tr_idx], dc["min_freq"], dc["max_vocab"])
    ids_all, len_all = data.encode(tok_train, vocab, dc["max_len"])
    ids_te, len_te = data.encode(tok_test, vocab, dc["max_len"])
    arrays = {"train_ids": ids_all[tr_idx], "train_len": len_all[tr_idx], "train_y": train_labels[tr_idx],
              "train_index": tr_idx, "val_ids": ids_all[va_idx], "val_len": len_all[va_idx],
              "val_y": train_labels[va_idx], "val_index": va_idx, "test_ids": ids_te, "test_len": len_te}

    n_tok = np.array([len(t) for t in tok_train])
    unk = lambda ids: float((ids == data.UNK).sum() / max(1, (ids != data.PAD).sum()))  # noqa: E731
    stats = {
        "stopwords_removed": len(stopwords), "negations_kept": sorted(data.NEGATIONS),
        "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(test_texts),
        "vocab_size": len(vocab), "tokens_per_review_mean": float(n_tok.mean()),
        "tokens_per_review_median": float(np.median(n_tok)), "tokens_per_review_p95": float(np.percentile(n_tok, 95)),
        "truncated_share": float((n_tok > dc["max_len"]).mean()),
        "empty_after_preprocessing_train": int((n_tok == 0).sum()),
        "empty_after_preprocessing_test": int(sum(1 for t in tok_test if not t)),
        "unk_rate_val": unk(arrays["val_ids"]), "unk_rate_test": unk(ids_te),
        "examples": [{"raw": train_texts[i][:300], "tokens": tok_train[i][:40]} for i in range(5)],
        "seconds": time.perf_counter() - t0,
    }
    np.savez_compressed(cache, **arrays)
    (proc / "vocab.json").write_text(json.dumps(vocab))
    (proc / "preprocessing_stats.json").write_text(json.dumps(stats, indent=2))
    log.info(f"preprocessing: {json.dumps({k: v for k, v in stats.items() if k != 'examples'})}")
    return arrays, vocab


# ---------------- training ----------------
def batches(n, bs, shuffle, gen=None):
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


def run_epoch(model, ids, lens, y, bs, device, opt=None, grad_clip=None, gen=None):
    """One pass over (ids, lens, y). Trains if opt is given. Returns (mean loss, accuracy, probs)."""
    train = opt is not None
    model.train(train)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total, correct, probs = 0.0, 0, []
    with torch.set_grad_enabled(train):
        for b in batches(len(y), bs, shuffle=train, gen=gen):
            L = lens[b].to(device)
            T = int(L.max())
            x = ids[b, :T].to(device).long()  # trim padding to the longest review in the batch
            t = y[b].to(device).float()
            logits = model(x, L)
            loss = loss_fn(logits, t)
            if train:
                opt.zero_grad(set_to_none=True)
                (loss / len(b)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            total += loss.item()
            p = torch.sigmoid(logits.detach().float())
            correct += ((p >= 0.5).long() == t.long()).sum().item()
            if not train:
                probs.append(p.cpu())
    return total / len(y), correct / len(y), (torch.cat(probs).numpy() if probs else None)


def train_model(name, mcfg, cfg, arrays, vocab_size, device, log):
    tc = cfg["train"]
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    keep_on_gpu = device.type == "cuda"
    T = lambda k: torch.from_numpy(arrays[k]).to(device if keep_on_gpu else "cpu")  # noqa: E731
    tr = (T("train_ids"), T("train_len"), T("train_y"))
    va = (T("val_ids"), T("val_len"), T("val_y"))
    te = (T("test_ids"), T("test_len"))

    model = build_model(mcfg, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=mcfg["lr"])
    log.info(f"[{name}] {mcfg} | parameters={n_params:,}")

    utils.reset_peak_memory(device)
    history, best, bad, train_secs = [], (math.inf, None, 0), 0, 0.0
    for epoch in range(1, tc["epochs"] + 1):
        utils.sync(device)
        t0 = time.perf_counter()
        tr_loss, tr_acc, _ = run_epoch(model, *tr, tc["batch_size"], device, opt, tc["grad_clip"], gen)
        utils.sync(device)
        secs = time.perf_counter() - t0
        train_secs += secs
        va_loss, va_acc, _ = run_epoch(model, *va, tc["batch_size"] * 4, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
                        "val_acc": va_acc, "epoch_seconds": secs})
        log.info(f"[{name}] epoch {epoch} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} "
                 f"acc {va_acc:.4f} | {secs:.0f}s ({len(tr[2]) / secs:,.0f} ex/s)")
        if not math.isfinite(tr_loss):
            log.warning(f"[{name}] non-finite training loss - stopping")
            break
        if va_loss < best[0]:
            best, bad = (va_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch), 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                log.info(f"[{name}] early stopping: no val improvement for {bad} epochs")
                break
    model.load_state_dict(best[1])
    peak = utils.peak_memory_mb(device)

    utils.sync(device)
    t0 = time.perf_counter()
    model.eval()
    probs = []
    with torch.no_grad():
        for b in batches(len(te[1]), tc["batch_size"] * 4, shuffle=False):
            L = te[1][b].to(device)
            probs.append(torch.sigmoid(model(te[0][b, :int(L.max())].to(device).long(), L).float()).cpu())
    utils.sync(device)
    infer_secs = time.perf_counter() - t0
    test_probs = torch.cat(probs).numpy()

    ckpt = MEMBER_DIR / "checkpoints" / f"{name}_best.pt"
    torch.save({"model_state": model.state_dict(), "model_cfg": mcfg, "vocab_size": vocab_size,
                "best_epoch": best[2], "val_loss": best[0]}, ckpt)
    epochs_run = len(history)
    info = {"parameters": n_params, "epochs_run": epochs_run, "best_epoch": best[2], "best_val_loss": best[0],
            "best_val_acc": history[best[2] - 1]["val_acc"], "train_time_s": train_secs,
            "train_examples_per_s": epochs_run * len(tr[2]) / train_secs,
            "inference_examples_per_s": len(test_probs) / infer_secs, **peak, "checkpoint": utils.rel(ckpt)}
    return test_probs, history, info


# ---------------- error candidates for the manual review (brief 2.2.4) ----------------
def error_candidates(name, y, p, test_texts, slice_labels, slice_rows, k=10):
    pred = (p >= 0.5).astype(int)
    wrong = np.flatnonzero(pred != y)
    conf = np.abs(p - 0.5)
    fp = wrong[y[wrong] == 0]
    fn = wrong[y[wrong] == 1]
    picks = {
        "confident_false_positive": fp[np.argsort(-p[fp])][:k],
        "confident_false_negative": fn[np.argsort(p[fn])][:k],
        "near_threshold": wrong[np.argsort(conf[wrong])][:k],
    }
    # slice-specific: errors from this model's worst slice (highest error rate), most confident first
    worst = max(slice_rows, key=lambda r: r["error_rate"])
    used = set(np.concatenate(list(picks.values())).tolist())
    in_slice = [i for i in wrong[np.argsort(-conf[wrong])] if slice_labels[worst["slice"]][i] == worst["value"] and i not in used]
    picks[f"slice:{worst['slice']}={worst['value']}"] = np.array(in_slice[:k], dtype=int)
    rows = []
    for cat, idx in picks.items():
        for i in idx:
            rows.append({"model": name, "category": cat, "test_index": int(i), "true": int(y[i]), "prob_positive": float(p[i]),
                         **{s: slice_labels[s][i] for s in slice_labels}, "text": test_texts[i][:1500]})
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))  # union of keys, first-seen order
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task2_sentiment/kruthika/src/config.yaml")
    ap.add_argument("--epochs", type=int, help="override train.epochs for all models")
    ap.add_argument("--force", action="store_true", help="retrain models that already have results")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    rid = utils.run_id("sentiment")
    log, log_path = utils.get_logger(rid)
    utils.set_seed(cfg["seed"])
    device = utils.pick_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    env = utils.environment_info(device)
    log.info(f"run_id={rid} config={args.config} force={args.force}")
    log.info(f"python={env['python']} torch={env['torch']} device={device} ({env['device_name']}) "
             f"git={env['git_commit']} dirty={env['git_dirty']}")
    log.info(f"config: {json.dumps(cfg)}")

    out = MEMBER_DIR / "outputs"
    for d in (out / "eda", MEMBER_DIR / "checkpoints"):
        d.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    data_dir = utils.repo_path(cfg["data"]["data_dir"])
    if not data_dir.is_dir():
        download_yelp_polarity()
    from datasets import load_from_disk
    ds = load_from_disk(str(data_dir))
    train_texts, test_texts = ds["train"]["text"], ds["test"]["text"]
    train_labels = np.array(ds["train"]["label"], dtype=np.int64)
    test_y = np.array(ds["test"]["label"], dtype=np.int64)
    eda(train_texts, train_labels, test_texts, test_y, out / "eda", log)
    arrays, vocab = prepare(cfg, train_texts, train_labels, test_texts, log)
    test_slices = data.slices(test_texts)

    # ---- train / load each model ----
    probs, histories, infos = {}, {}, {}
    for name, mcfg in cfg["models"].items():
        mdir = out / name
        mdir.mkdir(exist_ok=True)
        done = (mdir / "test_probs.npy").exists() and (mdir / "info.json").exists() \
            and (MEMBER_DIR / "checkpoints" / f"{name}_best.pt").exists()
        if done and not args.force:
            log.info(f"[{name}] already trained - loading saved results (use --force to retrain)")
            probs[name] = np.load(mdir / "test_probs.npy")
            histories[name] = json.loads((mdir / "history.json").read_text())
            infos[name] = json.loads((mdir / "info.json").read_text())
            continue
        p, h, info = train_model(name, mcfg, cfg, arrays, len(vocab), device, log)
        np.save(mdir / "test_probs.npy", p)
        (mdir / "history.json").write_text(json.dumps(h, indent=2))
        (mdir / "info.json").write_text(json.dumps(info, indent=2))
        probs[name], histories[name], infos[name] = p, h, info

    # ---- evaluate & compare ----
    ec = cfg["eval"]
    names = list(cfg["models"])
    base = names[0]
    report, comparison, slice_all, mcn_rows, errors = [], [], [], [], []
    for name in names:
        p = probs[name]
        m = ev.classification_metrics(test_y, p, ec["ece_bins"])
        m.update(ev.bootstrap_ci(test_y, p, ec["bootstrap"], seed=cfg["seed"]))
        info = infos[name]
        m.update({k: info[k] for k in ("parameters", "train_time_s", "train_examples_per_s",
                                        "inference_examples_per_s", "epochs_run", "best_epoch")})
        m.update({k: v for k, v in info.items() if k.startswith("peak_") or k.startswith("mps_")})
        m["device"] = env["device_name"]
        if name != base:
            t = ev.mcnemar(test_y, probs[base], p)
            mcn_rows.append({"baseline": base, "experimental": name, **t})
            m.update({f"mcnemar_vs_baseline_{k}": v for k, v in t.items()})
        srows = ev.slice_metrics(test_y, p, test_slices)
        for r in srows:
            slice_all.append({"model": name, **r})
            m[f"slice[{r['slice']}={r['value']}]_f1_macro"] = r["f1_macro"]
            m[f"slice[{r['slice']}={r['value']}]_error_rate"] = r["error_rate"]
        ev.plot_confusion(test_y, p, name, out / name / "confusion_matrix.png")
        errors += error_candidates(name, test_y, p, test_texts, test_slices, srows)
        comparison.append({"model": name, **{k: v for k, v in m.items() if not k.startswith("slice[")}})
        report += [{"model": name, "metric": k, "value": v} for k, v in m.items()]
        log.info(f"[{name}] TEST acc {m['accuracy']:.4f} [{m['accuracy_ci_low']:.4f}, {m['accuracy_ci_high']:.4f}] "
                 f"macroF1 {m['f1_macro']:.4f} MCC {m['mcc']:.4f} ROC-AUC {m['roc_auc']:.4f} "
                 f"PR-AUC {m['pr_auc']:.4f} Brier {m['brier']:.4f} ECE {m['ece']:.4f}")
    for r in mcn_rows:
        log.info(f"McNemar {r['baseline']} vs {r['experimental']}: b={r['b_base_only_correct']} "
                 f"c={r['c_exp_only_correct']} chi2={r['chi2']:.2f} p={r['p_value']:.3g}")

    write_csv(MEMBER_DIR / "metrics_report.csv", report)
    write_csv(out / "comparison.csv", comparison)
    write_csv(out / "slice_metrics.csv", slice_all)
    write_csv(out / "mcnemar.csv", mcn_rows)
    write_csv(out / "error_candidates.csv", errors)
    pred_rows = [{"test_index": i, "true": int(test_y[i]), **{s: test_slices[s][i] for s in test_slices},
                  **{f"prob_{n}": round(float(probs[n][i]), 6) for n in names}} for i in range(len(test_y))]
    write_csv(out / "test_predictions.csv", pred_rows)
    ev.plot_curves(test_y, probs, out / "roc_pr_calibration.png")
    ev.plot_history(histories, out / "training_curves.png")

    checkpoints = [MEMBER_DIR / "checkpoints" / f"{n}_best.pt" for n in names]
    manifest = utils.write_manifest(rid, cfg, device, log_path, checkpoints=checkpoints,
                                    metrics={r["model"]: r for r in comparison}, extra={"kind": "train"})
    log.info(f"metrics  -> {utils.rel(MEMBER_DIR / 'metrics_report.csv')}")
    log.info(f"raw log  -> {utils.rel(log_path)}")
    log.info(f"manifest -> {utils.rel(manifest)}")


if __name__ == "__main__":
    main()
