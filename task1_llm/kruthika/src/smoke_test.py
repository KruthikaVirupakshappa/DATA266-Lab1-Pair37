"""Task 1 smoke test: a tiny end-to-end run that checks the whole pipeline works.

It does NOT produce reportable results - it only proves that data -> model -> training ->
checkpoint -> generation runs correctly on a small subset, and that logs/manifests are written.

Usage (from the repo root):
    python task1_llm/kruthika/src/smoke_test.py

It uses config.yaml with the small values from its `smoke:` section. Only a raw log and
a manifest are written (to reproducibility/); the test checkpoint goes to a temp folder.

Exit code 0 = every check passed.
"""
import argparse
import json
import math
import sys
import tempfile
import time
import traceback
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


class Checks:
    """Runs named checks, logging PASS / FAIL / TODO (NotImplementedError) for each."""

    def __init__(self, log):
        self.log = log
        self.results = []

    def __call__(self, name, fn):
        try:
            out = fn()
            self.results.append((name, "PASS", ""))
            self.log.info(f"[PASS] {name}")
            return out
        except NotImplementedError as e:
            self.results.append((name, "TODO", str(e)))
            self.log.warning(f"[TODO] {name}: {e}")
        except Exception as e:
            self.results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
            self.log.error(f"[FAIL] {name}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise _Stop

    def summary(self):
        self.log.info("=" * 60)
        for name, status, msg in self.results:
            self.log.info(f"{status:4s}  {name}" + (f"  -- {msg}" if msg and status != "PASS" else ""))
        ok = bool(self.results) and all(s == "PASS" for _, s, _ in self.results)
        self.log.info(f"SMOKE TEST {'PASSED' if ok else 'DID NOT PASS'}")
        return ok


class _Stop(Exception):
    """Later checks depend on earlier ones, so stop at the first non-PASS."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task1_llm/kruthika/src/config.yaml")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    smoke = cfg.pop("smoke")
    for section, values in smoke.items():  # shrink the real config for a quick run
        cfg[section] = {**cfg[section], **values}
    cfg["run_name"] = "smoke"
    rid = utils.run_id(cfg["run_name"])
    log, log_path = utils.get_logger(rid)
    utils.set_seed(cfg["seed"])
    device = utils.pick_device(cfg.get("device", "auto"))
    env = utils.environment_info(device)
    log.info(f"run_id={rid}  config={args.config}")
    log.info(f"python={env['python']} torch={env['torch']} device={device} ({env['device_name']}) "
             f"git={env['git_commit']} dirty={env['git_dirty']}")
    log.info(f"config: {json.dumps(cfg)}")

    check = Checks(log)
    out = {}
    try:
        run_checks(cfg, device, check, log, out)
    except _Stop:
        pass

    ok = check.summary()
    out["checks"] = {n: s for n, s, _ in check.results}
    manifest = utils.write_manifest(rid, cfg, device, log_path, metrics=out,
                                    extra={"kind": "smoke_test", "passed": ok})
    log.info(f"raw log  -> {utils.rel(log_path)}")
    log.info(f"manifest -> {utils.rel(manifest)}")
    sys.exit(0 if ok else 1)


def run_checks(cfg, device, check, log, res):
    """Runs every check in order, filling `res` with metrics as it goes."""
    dc, tc, gc = cfg["data"], cfg["train"], cfg["generate"]
    T, B = dc["block_size"], tc["batch_size"]
    gen = torch.Generator().manual_seed(cfg["seed"])

    # ---------------- 1. data ----------------
    def load():
        path = utils.repo_path(dc["data_dir"])
        if not path.is_dir():
            download_tinystories()
        tr, va = data.load_stories(path, dc["n_train"], dc["n_val"], cfg["seed"])
        assert len(tr) == dc["n_train"], f"expected {dc['n_train']} train stories, got {len(tr)}"
        assert len(va) == dc["n_val"], f"expected {dc['n_val']} val stories, got {len(va)}"
        assert all(isinstance(s, str) and s for s in tr[:50] + va[:50]), "stories must be non-empty str"
        overlap = len(set(tr) & set(va))
        if overlap:
            log.warning(f"{overlap} identical stories appear in both train and val")
        return tr, va
    train_texts, val_texts = check("data.load_stories: split sizes", load)

    def vocab():
        c2i, i2c = data.build_vocab(train_texts)
        V = len(c2i)
        assert sorted(c2i.values()) == list(range(V)), "indices must be 0..V-1 with no gaps"
        assert all(i2c[i] == c for c, i in c2i.items()), "idx_to_char must invert char_to_idx"
        assert all(len(c) == 1 for c in c2i), "character-level vocab: every key is one char"
        sample = train_texts[0]
        assert data.decode(data.encode(sample, c2i), i2c) == sample, "decode(encode(x)) != x"
        log.info(f"vocab_size={V}  chars={''.join(sorted(c2i))!r}")
        return c2i, i2c
    c2i, i2c = check("data.build_vocab/encode/decode: round trip", vocab)
    V = len(c2i)
    res["vocab_size"] = V

    def encode_splits():
        tr = data.encode_split(train_texts, c2i)
        va = data.encode_split(val_texts, c2i)  # must cope with chars unseen in train
        for name, t in (("train", tr), ("val", va)):
            assert isinstance(t, torch.Tensor) and t.dim() == 1 and t.dtype == torch.long, \
                f"{name}: expected 1-D torch.long tensor, got {type(t)}"
            assert 0 <= int(t.min()) and int(t.max()) < V, f"{name}: ids out of range [0, {V})"
        log.info(f"train chars={len(tr):,}  val chars={len(va):,}")
        return tr, va
    train_ids, val_ids = check("data.encode_split: 1-D id tensors", encode_splits)
    res["train_chars"], res["val_chars"] = len(train_ids), len(val_ids)

    def batch():
        x, y = data.get_batch(train_ids, T, B, device, generator=gen)
        assert x.shape == (B, T) and y.shape == (B, T), f"got x{tuple(x.shape)} y{tuple(y.shape)}"
        assert x.dtype == torch.long and x.device.type == device.type
        assert torch.equal(x[:, 1:], y[:, :-1]), "targets must be inputs shifted left by one"
        return x, y
    x, y = check("data.get_batch: shapes + next-char targets", batch)

    # ---------------- 2. model ----------------
    model_cfg = {**(cfg.get("model") or {}), "vocab_size": V, "block_size": T}

    def build():
        m = GPT(model_cfg).to(device)
        n = metrics.count_parameters(m)
        assert n > 0, "model has no trainable parameters"
        log.info(f"parameters={n:,}  model_cfg={model_cfg}")
        return m, n
    model, res["n_params"] = check("model: construct GPT(cfg)", build)

    def forward():
        model.train()
        logits, loss = model(x, y)
        assert logits.shape == (B, T, V), f"logits {tuple(logits.shape)} != {(B, T, V)}"
        assert loss is not None and loss.dim() == 0 and torch.isfinite(loss), f"bad loss {loss}"
        _, no_loss = model(x)
        assert no_loss is None, "forward(idx) without targets should return loss=None"
        init, uniform = loss.item(), math.log(V)
        log.info(f"initial loss={init:.4f}  (uniform-guess loss ln(V)={uniform:.4f})")
        if init > uniform + 1.0:
            log.warning("initial loss is far above ln(V) - check output-layer init / logits scale")
        return init
    res["initial_loss"] = check("model.forward: logits shape + finite loss", forward)

    def causal():
        model.eval()
        with torch.no_grad():
            x0 = x[:1].clone()
            t = T // 2
            x1 = x0.clone()
            x1[0, t] = (x1[0, t] + 1) % V
            l0, _ = model(x0)
            l1, _ = model(x1)
            leak = (l0[:, :t] - l1[:, :t]).abs().max().item()
            assert leak < 1e-4, f"changing token {t} changed logits at earlier positions (max diff {leak:.2e}) - future is leaking"
            assert not torch.allclose(l0[:, t:], l1[:, t:]), "changing a token had no effect at all"
            short, _ = model(x0[:, : T // 4])
            assert short.shape == (1, T // 4, V), "model must accept sequences shorter than block_size"
    check("model: causal mask (no future leakage) + variable length", causal)

    # ---------------- 3. training ----------------
    def train():
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"])  # smoke only: constant LR
        losses, gnorms, bad = [], [], 0
        utils.reset_peak_memory(device)
        utils.sync(device)
        t0 = time.perf_counter()
        for step in range(1, tc["steps"] + 1):
            xb, yb = data.get_batch(train_ids, T, B, device, generator=gen)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = metrics.grad_norm(model)
            if not (math.isfinite(loss.item()) and math.isfinite(gn)):
                bad += 1
            opt.step()
            losses.append(loss.item())
            gnorms.append(gn)
            if step % tc["log_every"] == 0 or step == 1:
                log.info(f"step {step:5d} | loss {loss.item():.4f} | grad_norm {gn:.3f}")
        utils.sync(device)
        secs = time.perf_counter() - t0
        k = max(1, min(10, len(losses) // 4))
        first, last = sum(losses[:k]) / k, sum(losses[-k:]) / k
        assert bad == 0, f"{bad} steps had NaN/inf loss or gradients"
        assert last < 0.95 * first, f"loss did not go down ({first:.3f} -> {last:.3f})"
        return {"train_steps": tc["steps"], "train_time_s": secs,
                "train_tokens_per_s": tc["steps"] * B * T / secs,
                "loss_first": first, "loss_last": last, "nan_inf_steps": bad,
                "grad_norm_max": max(gnorms), "grad_norm_mean": sum(gnorms) / len(gnorms),
                **utils.peak_memory_mb(device)}
    res.update(check("train: loss decreases, no NaN/inf", train))

    def evaluate():
        model.eval()
        out = {}
        with torch.no_grad():
            for split, ids in (("train", train_ids), ("val", val_ids)):
                ls, accs = [], []
                for _ in range(tc["eval_batches"]):
                    xb, yb = data.get_batch(ids, T, B, device, generator=gen)
                    logits, loss = model(xb, yb)
                    ls.append(loss.item())
                    accs.append(metrics.top1_accuracy(logits, yb))
                out[f"{split}_loss"] = sum(ls) / len(ls)
                out[f"{split}_top1_acc"] = sum(accs) / len(accs)
        out["val_perplexity"] = metrics.perplexity(out["val_loss"])
        out["val_bpc"] = metrics.bits_per_char(out["val_loss"])
        out["generalization_gap"] = out["val_loss"] - out["train_loss"]
        log.info(" | ".join(f"{k} {v:.4f}" for k, v in out.items()))
        return out
    res.update(check("eval: val loss / perplexity / bpc / top-1", evaluate))

    # ---------------- 4. checkpoint ----------------
    def checkpoint():
        path = Path(tempfile.mkdtemp()) / "smoke.pt"
        torch.save({"model_state": model.state_dict(), "model_cfg": model_cfg,
                    "char_to_idx": c2i, "config": cfg}, path)
        ck = torch.load(path, map_location=device)
        m2 = GPT(ck["model_cfg"]).to(device)
        m2.load_state_dict(ck["model_state"])
        m2.eval()
        model.eval()
        with torch.no_grad():
            diff = (model(x)[0] - m2(x)[0]).abs().max().item()
        assert diff < 1e-5, f"reloaded model differs (max logit diff {diff:.2e})"
        log.info(f"checkpoint save/reload OK ({path.stat().st_size / 2**20:.2f} MB)")
        path.unlink()
    check("checkpoint: save + reload identical", checkpoint)

    # ---------------- 5. generation ----------------
    def generate():
        model.eval()
        prompt = torch.tensor([data.encode(gc["prompt"], c2i)], dtype=torch.long, device=device)
        texts, new_tokens = [], 0
        utils.sync(device)
        t0 = time.perf_counter()
        for _ in range(gc["n_samples"]):
            with torch.no_grad():
                o = model.generate(prompt, gc["max_new_tokens"], temperature=gc["temperature"])
            assert o.shape == (1, prompt.shape[1] + gc["max_new_tokens"]), \
                f"generate returned {tuple(o.shape)}; expected prompt + max_new_tokens"
            assert torch.equal(o[:, :prompt.shape[1]], prompt), "generate must keep the prompt as a prefix"
            texts.append(data.decode(o[0].tolist(), i2c))
            new_tokens += gc["max_new_tokens"]
        utils.sync(device)
        secs = time.perf_counter() - t0
        for i, t in enumerate(texts):
            log.info(f"sample {i}: {t!r}")
        return {"gen_tokens_per_s": new_tokens / secs,
                **{f"distinct_{n}": metrics.distinct_n(texts, n) for n in (1, 2, 3)},
                "repeated_4gram_rate": metrics.repeated_ngram_rate(texts, 4)}
    res.update(check("generate: prompt continuation (longer than block_size)", generate))


if __name__ == "__main__":
    main()
