"""Task 1.3 - train the GPT on TinyStories, then evaluate and generate.

Usage (from the repo root):
    python task1_llm/kruthika/src/train.py            # settings from src/config.yaml
    python task1_llm/kruthika/src/train.py --resume   # continue after a disconnect

One "epoch" = as many randomly sampled training characters as the training split
contains (steps_per_epoch = train_chars // (batch_size * block_size)).

Writes (paths relative to task1_llm/kruthika/):
    checkpoints/<run>_best.pt      model weights with the lowest val loss (small; commit this)
    checkpoints/<run>_resume.pt    model + optimizer + progress, for --resume (large; git-ignored)
    outputs/<run>/                 history CSV, loss curves, samples, metrics JSON
    metrics_report.csv             every Task 1 metric in one file
plus a raw log and a manifest in reproducibility/.
"""
import argparse
import contextlib
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(SRC.parents[1] / "scripts"))  # task1_llm/scripts, for the downloader

import data  # noqa: E402
import metrics  # noqa: E402
import utils  # noqa: E402
from download_tinystories import download_tinystories  # noqa: E402
from model import GPT  # noqa: E402

MEMBER_DIR = utils.REPO_ROOT / "task1_llm" / utils.MEMBER


def lr_at(step, tc, total_steps):
    """Linear warm-up from 0 to lr, then cosine decay down to min_lr."""
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    progress = (step - tc["warmup_steps"]) / max(1, total_steps - tc["warmup_steps"])
    return tc["min_lr"] + 0.5 * (tc["lr"] - tc["min_lr"]) * (1 + math.cos(math.pi * min(1.0, progress)))


def make_optimizer(model, tc):
    """AdamW; weight decay only on matrices (Linear/Embedding weights), not on biases or LayerNorm."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": tc["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    fused = {"fused": True} if next(model.parameters()).is_cuda else {}
    return torch.optim.AdamW(groups, lr=tc["lr"], betas=tuple(tc["betas"]), **fused)


def autocast(device, precision):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def estimate_loss(model, ids, T, B, n_batches, device, precision, gen):
    """Mean loss and top-1 accuracy over n_batches random windows (quick check during training)."""
    model.eval()
    losses, accs = [], []
    for _ in range(n_batches):
        x, y = data.get_batch(ids, T, B, device, generator=gen)
        with autocast(device, precision):
            logits, loss = model(x, y)
        losses.append(loss.item())
        accs.append(metrics.top1_accuracy(logits, y))
    model.train()
    return sum(losses) / len(losses), sum(accs) / len(accs)


@torch.no_grad()
def full_eval(model, ids, T, B, device, precision, max_windows=None):
    """Loss and top-1 accuracy over consecutive non-overlapping windows covering the split."""
    model.eval()
    n = (len(ids) - 1) // T
    if max_windows:
        n = min(n, max_windows)
    xs = ids[: n * T].view(n, T)
    ys = ids[1: n * T + 1].view(n, T)
    loss_sum, correct = 0.0, 0
    for i in range(0, n, B):
        x, y = xs[i:i + B].to(device), ys[i:i + B].to(device)
        with autocast(device, precision):
            logits, loss = model(x, y)
        loss_sum += loss.item() * y.numel()
        correct += (logits.argmax(-1) == y).sum().item()
    model.train()
    return loss_sum / (n * T), correct / (n * T)


def save_plots(history, evals, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, [h["train_loss"] for h in history], alpha=0.35, label="train (per logged step)")
    ax.plot([e["step"] for e in evals], [e["train_loss"] for e in evals], "o-", label="train (eval batches)")
    ax.plot([e["step"] for e in evals], [e["val_loss"] for e in evals], "o-", label="validation")
    ax.set(xlabel="step", ylabel="cross-entropy loss (nats)", title="Training and validation loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(steps, [h["lr"] for h in history])
    a1.set(ylabel="learning rate", title="LR schedule and gradient norm")
    a2.plot(steps, [h["grad_norm"] for h in history])
    a2.set(xlabel="step", ylabel="grad norm (pre-clip)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "lr_gradnorm.png", dpi=150)
    plt.close(fig)


def generate_samples(model, gc, c2i, i2c, device, log):
    """Sample from each prompt with temperature sampling and greedy decoding; time generation."""
    model.eval()
    results = {"sampled": [], "greedy": []}
    new_tokens, secs = 0, 0.0
    for prompt in gc["prompts"]:
        idx = torch.tensor([data.encode(prompt, c2i)], dtype=torch.long, device=device)
        runs = [("sampled", gc["temperature"])] * gc["samples_per_prompt"] + [("greedy", 0)]
        for kind, temp in runs:
            utils.sync(device)
            t0 = time.perf_counter()
            out = model.generate(idx, gc["max_new_tokens"], temperature=temp, top_k=gc.get("top_k"))
            utils.sync(device)
            secs += time.perf_counter() - t0
            new_tokens += gc["max_new_tokens"]
            text = data.decode(out[0].tolist(), i2c)
            results[kind].append({"prompt": prompt, "text": text})
            log.info(f"[{kind}] {text!r}")
    model.train()
    return results, new_tokens / secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task1_llm/kruthika/src/config.yaml")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>_resume.pt")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    dc, mc, tc, gc = cfg["data"], cfg["model"], cfg["train"], cfg["generate"]
    run = cfg["run_name"]
    T, B = dc["block_size"], tc["batch_size"]
    rid = utils.run_id(run)
    log, log_path = utils.get_logger(rid)
    utils.set_seed(cfg["seed"])
    device = utils.pick_device(cfg.get("device", "auto"))
    precision = tc.get("precision", "fp32") if device.type == "cuda" else "fp32"
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        precision = "fp32"  # older GPUs have no bf16 support
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    env = utils.environment_info(device)
    log.info(f"run_id={rid} config={args.config} resume={args.resume}")
    log.info(f"python={env['python']} torch={env['torch']} device={device} ({env['device_name']}) "
             f"precision={precision} git={env['git_commit']} dirty={env['git_dirty']}")
    log.info(f"config: {json.dumps(cfg)}")

    out_dir = MEMBER_DIR / "outputs" / run
    ckpt_dir = MEMBER_DIR / "checkpoints"
    proc_dir = MEMBER_DIR / "data_processed"
    for d in (out_dir, ckpt_dir, proc_dir):
        d.mkdir(parents=True, exist_ok=True)
    best_path, resume_path = ckpt_dir / f"{run}_best.pt", ckpt_dir / f"{run}_resume.pt"

    # ---------------- data ----------------
    t0 = time.perf_counter()
    data_dir = utils.repo_path(dc["data_dir"])
    if not data_dir.is_dir():
        download_tinystories()
    train_texts, val_texts = data.load_stories(data_dir, dc["n_train"], dc["n_val"], cfg["seed"])
    c2i, i2c = data.build_vocab(train_texts)
    V = len(c2i)
    train_ids = data.encode_split(train_texts, c2i)
    val_ids = data.encode_split(val_texts, c2i)
    if device.type == "cuda":  # whole dataset fits easily in GPU memory -> no per-step copies
        train_ids, val_ids = train_ids.to(device), val_ids.to(device)
    (proc_dir / "vocab.json").write_text(json.dumps({"char_to_idx": c2i}, indent=1))
    (proc_dir / "split_stats.json").write_text(json.dumps({
        "data_dir": dc["data_dir"], "seed": cfg["seed"],
        "n_train_stories": len(train_texts), "n_val_stories": len(val_texts),
        "train_chars": len(train_ids), "val_chars": len(val_ids), "vocab_size": V,
    }, indent=1))
    log.info(f"data: {len(train_texts)} train / {len(val_texts)} val stories | "
             f"{len(train_ids):,} / {len(val_ids):,} chars | vocab={V} | {time.perf_counter() - t0:.1f}s")

    # ---------------- model ----------------
    model_cfg = {**mc, "vocab_size": V, "block_size": T}
    model = GPT(model_cfg).to(device)
    n_params = metrics.count_parameters(model)
    log.info(f"model: {model_cfg} | parameters={n_params:,}")
    opt = make_optimizer(model, tc)

    steps_per_epoch = len(train_ids) // (B * T)
    total_steps = tc["epochs"] * steps_per_epoch
    log.info(f"schedule: {tc['epochs']} epochs x {steps_per_epoch} steps = {total_steps} steps "
             f"({total_steps * B * T / 1e6:.0f}M tokens), warmup={tc['warmup_steps']}")

    gen = torch.Generator().manual_seed(cfg["seed"])
    eval_gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    step, history, evals, best_val = 0, [], [], float("inf")
    nan_steps, spikes, train_secs = 0, 0, 0.0
    if args.resume and resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["opt_state"])
        gen.set_state(ck["gen_state"].cpu())  # RNG state must stay on CPU
        step, history, evals, best_val = ck["step"], ck["history"], ck["evals"], ck["best_val"]
        nan_steps, spikes, train_secs = ck["nan_steps"], ck["spikes"], ck["train_secs"]
        log.info(f"resumed from {utils.rel(resume_path)} at step {step}")

    train_model = torch.compile(model) if (tc.get("compile") and device.type == "cuda") else model

    def save_resume():
        torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(),
                    "gen_state": gen.get_state(), "step": step, "history": history, "evals": evals,
                    "best_val": best_val, "nan_steps": nan_steps, "spikes": spikes,
                    "train_secs": train_secs, "model_cfg": model_cfg, "config": cfg}, resume_path)

    # ---------------- training loop ----------------
    utils.reset_peak_memory(device)
    ema = None
    train_model.train()
    utils.sync(device)
    t_last, tok_last = time.perf_counter(), 0
    while step < total_steps:
        lr = lr_at(step, tc, total_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = data.get_batch(train_ids, T, B, device, generator=gen)
        with autocast(device, precision):
            _, loss = train_model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]).item()
        loss_val = loss.item()
        if not (math.isfinite(loss_val) and math.isfinite(gnorm)):
            nan_steps += 1  # skip the update rather than corrupt the weights
            log.warning(f"step {step}: non-finite loss/grad (loss={loss_val}, grad={gnorm}) - update skipped")
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
            if ema is not None and step > tc["warmup_steps"] and loss_val > 1.3 * ema:
                spikes += 1
                log.warning(f"step {step}: loss spike {loss_val:.4f} vs running mean {ema:.4f}")
            ema = loss_val if ema is None else 0.98 * ema + 0.02 * loss_val
        step += 1
        tok_last += B * T

        if step % tc["log_every"] == 0 or step == total_steps:
            utils.sync(device)
            dt = time.perf_counter() - t_last
            train_secs += dt
            tps = tok_last / dt
            history.append({"step": step, "epoch": step / steps_per_epoch, "lr": lr,
                            "train_loss": loss_val, "grad_norm": gnorm, "tokens_per_s": tps})
            log.info(f"step {step:6d}/{total_steps} | ep {step / steps_per_epoch:5.2f} | loss {loss_val:.4f} "
                     f"| lr {lr:.2e} | grad {gnorm:.3f} | {tps / 1e3:.0f}K tok/s")
            t_last, tok_last = time.perf_counter(), 0

        if step % tc["eval_every"] == 0 or step == total_steps:
            tr_loss, tr_acc = estimate_loss(train_model, train_ids, T, B, tc["eval_batches"], device, precision, eval_gen)
            va_loss, va_acc = estimate_loss(train_model, val_ids, T, B, tc["eval_batches"], device, precision, eval_gen)
            evals.append({"step": step, "epoch": step / steps_per_epoch, "train_loss": tr_loss,
                          "val_loss": va_loss, "train_acc": tr_acc, "val_acc": va_acc})
            log.info(f"EVAL step {step} | train {tr_loss:.4f} | val {va_loss:.4f} | "
                     f"gap {va_loss - tr_loss:+.4f} | val acc {va_acc:.3f}")
            if va_loss < best_val:
                best_val = va_loss
                torch.save({"model_state": model.state_dict(), "model_cfg": model_cfg,
                            "char_to_idx": c2i, "step": step, "val_loss": va_loss, "config": cfg}, best_path)
                log.info(f"new best val {va_loss:.4f} -> {utils.rel(best_path)}")
            t_last = time.perf_counter()  # don't count eval time as training time

        if step % tc["ckpt_every"] == 0 or step == total_steps:
            save_resume()
            t_last = time.perf_counter()

    # ---------------- final evaluation (best checkpoint) ----------------
    peak_mem = utils.peak_memory_mb(device)
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    log.info(f"final evaluation uses best checkpoint (step {best['step']})")
    val_loss, val_acc = full_eval(model, val_ids, T, B, device, precision)
    val_windows = (len(val_ids) - 1) // T
    train_loss, train_acc = full_eval(model, train_ids, T, B, device, precision, max_windows=val_windows)
    samples, gen_tps = generate_samples(model, gc, c2i, i2c, device, log)
    sampled = [s["text"] for s in samples["sampled"]]
    greedy = [s["text"] for s in samples["greedy"]]
    losses = [h["train_loss"] for h in history]

    report = {
        "run_name": run,
        "checkpoint": utils.rel(best_path),
        "best_step": best["step"],
        "train_cross_entropy": train_loss,
        "val_cross_entropy": val_loss,
        "val_perplexity": metrics.perplexity(val_loss),
        "val_bits_per_char": metrics.bits_per_char(val_loss),
        "train_perplexity": metrics.perplexity(train_loss),
        "generalization_gap": val_loss - train_loss,
        "train_top1_acc": train_acc,
        "val_top1_acc": val_acc,
        "distinct_1": metrics.distinct_n(sampled, 1),
        "distinct_2": metrics.distinct_n(sampled, 2),
        "distinct_3": metrics.distinct_n(sampled, 3),
        "repeated_4gram_rate": metrics.repeated_ngram_rate(sampled, 4),
        "greedy_repeated_4gram_rate": metrics.repeated_ngram_rate(greedy, 4),
        "grad_norm_mean": sum(h["grad_norm"] for h in history) / len(history),
        "grad_norm_max": max(h["grad_norm"] for h in history),
        "loss_spikes": spikes,
        "nan_inf_steps": nan_steps,
        "final_logged_train_loss": losses[-1],
        "parameters": n_params,
        "train_tokens_per_s": total_steps * B * T / train_secs,
        "gen_tokens_per_s": gen_tps,
        "total_train_time_s": train_secs,
        "total_train_time_min": train_secs / 60,
        "total_steps": total_steps,
        "epochs": tc["epochs"],
        "tokens_seen": total_steps * B * T,
        **peak_mem,
        "device": env["device_name"],
    }
    for k, v in report.items():
        log.info(f"METRIC {k} = {v}")

    # ---------------- write outputs ----------------
    with open(out_dir / "train_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    with open(out_dir / "eval_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(evals[0]))
        w.writeheader()
        w.writerows(evals)
    save_plots(history, evals, out_dir)
    (out_dir / "samples.json").write_text(json.dumps(samples, indent=2))
    with open(out_dir / "samples.txt", "w") as f:
        for kind in ("sampled", "greedy"):
            for s in samples[kind]:
                f.write(f"=== {kind} | prompt: {s['prompt']!r} ===\n{s['text']}\n\n")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    with open(MEMBER_DIR / "metrics_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(report.items())

    manifest = utils.write_manifest(rid, cfg, device, log_path, checkpoints=[best_path], metrics=report,
                                    extra={"kind": "train", "outputs": utils.rel(out_dir)})
    log.info(f"outputs  -> {utils.rel(out_dir)}")
    log.info(f"metrics  -> {utils.rel(MEMBER_DIR / 'metrics_report.csv')}")
    log.info(f"raw log  -> {utils.rel(log_path)}")
    log.info(f"manifest -> {utils.rel(manifest)}")


if __name__ == "__main__":
    main()
