"""Task 3.1 - train a CycleGAN (Monet <-> photo) from scratch.

Usage (from the repo root):
    python task3_gan/kruthika/src/train.py                 # settings from src/config.yaml
    python task3_gan/kruthika/src/train.py --epochs 100    # change the number of epochs
    python task3_gan/kruthika/src/train.py --config task3_gan/kruthika/src/config_v2.yaml   # fine-tune run
Training resumes automatically from checkpoints/<run>_last.pt after a disconnect. A config with
`train.init_from` starts from another run's weights (fine-tuning) with a fresh optimiser and schedule.

Losses (A = Monet, B = photo):
    adversarial (LSGAN): G_BA(photo) should fool D_A, G_AB(Monet) should fool D_B
    cycle:    |G_AB(G_BA(b)) - b| + |G_BA(G_AB(a)) - a|      (keeps content)
    identity: |G_BA(a) - a| + |G_AB(b) - b|                  (keeps colours stable)
Every `eval_every` epochs both generators are scored exactly on the images the evaluation script
uses (first 300 photos -> Monet, all 300 Monets -> photo, after a JPEG round trip), and the
best-scoring pair (raw or EMA weights) is saved as checkpoints/<run>_best.pt.
"""
import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(SRC.parents[1] / "scripts"))

import data  # noqa: E402
import metrics  # noqa: E402
import utils  # noqa: E402
from download_kaggle_data import download_kaggle_data  # noqa: E402
from models import Discriminator, Generator, ImagePool, count_params, init_weights  # noqa: E402

MEMBER_DIR = utils.REPO_ROOT / "task3_gan" / utils.MEMBER


def grad_norm(params):
    """Global L2 gradient norm as a GPU tensor (max_norm=inf: measures without clipping)."""
    return torch.nn.utils.clip_grad_norm_(params, float("inf"))


@torch.no_grad()
def ema_update(ema, model, decay):
    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)


@torch.no_grad()
def translate(G, images_uint8, device, precision, batch_size=32):
    """uint8 [N,3,256,256] -> uint8 translations (on CPU)."""
    G.eval()
    out = []
    for i in range(0, len(images_uint8), batch_size):
        x = data.to_float(images_uint8[i:i + batch_size].to(device))
        with autocast(device, precision):
            y = G(x)
        out.append(data.to_uint8(y.float()).cpu())
    return torch.cat(out)


def autocast(device, precision):
    import contextlib
    if device.type == "cuda" and precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def save_samples(G_BA, G_AB, photos, monets, device, precision, path):
    """Grid: photo | photo->Monet | cycle back, and Monet | Monet->photo | cycle back."""
    from PIL import Image
    fa = translate(G_BA, photos, device, precision)
    ra = translate(G_AB, fa, device, precision)
    fb = translate(G_AB, monets, device, precision)
    rb = translate(G_BA, fb, device, precision)
    rows = [torch.cat([p, f, r], dim=2) for p, f, r in zip(photos.cpu(), fa, ra)]
    rows += [torch.cat([m, f, r], dim=2) for m, f, r in zip(monets.cpu(), fb, rb)]
    grid = torch.cat(rows, dim=1).permute(1, 2, 0).numpy()
    Image.fromarray(grid).save(path, quality=90)


def plot_history(hist, evals, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for k, lab in (("loss_G_gan", "G adversarial"), ("loss_D_A", "D_A (Monet)"), ("loss_D_B", "D_B (photo)")):
        ax[0, 0].plot(ep, [h[k] for h in hist], label=lab, alpha=0.8)
    ax[0, 0].set(title="Generator / discriminator losses", xlabel="epoch", ylabel="LSGAN loss")
    for k, lab in (("loss_cycle", "cycle (L1)"), ("loss_identity", "identity (L1)")):
        ax[0, 1].plot(ep, [h[k] for h in hist], label=lab)
    ax[0, 1].set(title="Cycle-consistency and identity losses", xlabel="epoch", ylabel="L1 (images in [-1, 1])")
    ax[1, 0].plot(ep, [h["grad_norm_G"] for h in hist], label="generators")
    ax[1, 0].plot(ep, [h["grad_norm_D"] for h in hist], label="discriminators")
    ax[1, 0].set(title="Gradient norms", xlabel="epoch", ylabel="L2 norm", yscale="log")
    if evals:
        for k, lab in (("fid_B2A_ema", "photo->Monet (EMA)"), ("fid_A2B_ema", "Monet->photo (EMA)"),
                       ("FID_ema", "average (EMA)"), ("FID_raw", "average (raw)")):
            ax[1, 1].plot([e["epoch"] for e in evals], [e[k] for e in evals], "o-", label=lab)
    ax[1, 1].set(title="FID during training (evaluation-script images)", xlabel="epoch", ylabel="FID")
    for a in ax.ravel():
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task3_gan/kruthika/src/config.yaml")
    ap.add_argument("--epochs", type=int, help="override train.epochs")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    tc, mc = cfg["train"], cfg["model"]
    run = cfg["run_name"]
    rid = utils.run_id(run)
    log, log_path = utils.get_logger(rid)
    utils.set_seed(cfg["seed"])
    device = utils.pick_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    env = utils.environment_info(device)
    log.info(f"run_id={rid} config={args.config}")
    log.info(f"python={env['python']} torch={env['torch']} device={device} ({env['device_name']}) "
             f"git={env['git_commit']} dirty={env['git_dirty']}")
    log.info(f"config: {json.dumps(cfg)}")

    out_dir = MEMBER_DIR / "outputs" / run
    ckpt_dir = MEMBER_DIR / "checkpoints"
    for d in (out_dir / "samples", ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / f"{run}_last.pt", ckpt_dir / f"{run}_best.pt"

    # ---- data ----
    data_dir = utils.repo_path(cfg["data"]["data_dir"])
    download_kaggle_data()
    t0 = time.perf_counter()
    monets, _ = data.load_images(data_dir / "monet_jpg")
    photos, photo_names = data.load_images(data_dir / "photo_jpg")
    log.info(f"data: {len(monets)} Monet + {len(photos)} photos loaded in {time.perf_counter() - t0:.0f}s")
    monets_dev, photos_dev = monets.to(device), photos.to(device)
    eval_photos = photos[:metrics.N_EVAL]  # the evaluation script uses the first 300 files by name
    sample_p, sample_m = photos[:4], monets[:4]

    # ---- models ----
    G_AB, G_BA = Generator(**mc).to(device), Generator(**mc).to(device)    # Monet->photo, photo->Monet
    D_A, D_B = Discriminator(**mc).to(device), Discriminator(**mc).to(device)  # judge Monet, judge photo
    for m in (G_AB, G_BA, D_A, D_B):
        m.apply(init_weights)
        m.to(memory_format=torch.channels_last)
    ema_AB, ema_BA = copy.deepcopy(G_AB).eval(), copy.deepcopy(G_BA).eval()
    params_G = list(G_AB.parameters()) + list(G_BA.parameters())
    params_D = list(D_A.parameters()) + list(D_B.parameters())
    opt_G = torch.optim.Adam(params_G, lr=tc["lr_g"], betas=tuple(tc["betas"]))
    opt_D = torch.optim.Adam(params_D, lr=tc["lr_d"], betas=tuple(tc["betas"]))
    n_params = {"G_AB": count_params(G_AB), "G_BA": count_params(G_BA), "D_A": count_params(D_A), "D_B": count_params(D_B)}
    log.info(f"parameters: {n_params} total={sum(n_params.values()):,}")

    iters_per_epoch = math.ceil(len(photos) / tc["batch_size"])
    total_iters = tc["epochs"] * iters_per_epoch
    decay_iter = int(tc["decay_start"] * total_iters)

    def lr_factor(it):  # constant, then linear decay to 0
        return 1.0 if it < decay_iter else max(0.0, 1 - (it - decay_iter) / max(1, total_iters - decay_iter))

    inception = metrics.InceptionFeatures(device)
    real_monet_f, real_photo_f = inception(monets), inception(eval_photos)
    pool_A, pool_B = ImagePool(tc["pool_size"]), ImagePool(tc["pool_size"])
    lam_c, lam_i = tc["lambda_cycle"], tc["lambda_identity"] * tc["lambda_cycle"]
    mse, l1 = F.mse_loss, F.l1_loss
    aug = lambda x: data.diff_augment(x, tc["diffaug"])  # noqa: E731

    it, hist, evals, best, nan_steps, train_secs = 0, [], [], {"score": math.inf}, 0, 0.0
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        for name, m in (("G_AB", G_AB), ("G_BA", G_BA), ("D_A", D_A), ("D_B", D_B), ("ema_AB", ema_AB), ("ema_BA", ema_BA)):
            m.load_state_dict(ck[name])
        opt_G.load_state_dict(ck["opt_G"])
        opt_D.load_state_dict(ck["opt_D"])
        it, hist, evals, best = ck["iter"], ck["hist"], ck["evals"], ck["best"]
        nan_steps, train_secs = ck["nan_steps"], ck["train_secs"]
        log.info(f"resumed from {utils.rel(last_path)} at iteration {it} (epoch {it / iters_per_epoch:.2f})")
    elif tc.get("init_from"):
        src = utils.repo_path(tc["init_from"])
        ck = torch.load(src, map_location=device, weights_only=False)
        for name, m in (("G_AB", G_AB), ("G_BA", G_BA), ("D_A", D_A), ("D_B", D_B), ("ema_AB", ema_AB), ("ema_BA", ema_BA)):
            m.load_state_dict(ck[name])
        log.info(f"initialised from {utils.rel(src)} (iteration {ck['iter']} of run "
                 f"{ck['config']['run_name']}); fresh optimisers and LR schedule")
    log.info(f"schedule: {tc['epochs']} epochs x {iters_per_epoch} iters = {total_iters} iterations, "
             f"batch {tc['batch_size']}, LR decay from iteration {decay_iter}")

    def save_last():
        torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "D_A": D_A.state_dict(),
                    "D_B": D_B.state_dict(), "ema_AB": ema_AB.state_dict(), "ema_BA": ema_BA.state_dict(),
                    "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(), "iter": it, "hist": hist,
                    "evals": evals, "best": best, "nan_steps": nan_steps, "train_secs": train_secs,
                    "config": cfg}, last_path)

    def evaluate(epoch):
        nonlocal best
        row = {"epoch": epoch, "iter": it}
        q = cfg["eval"]["jpg_quality"]
        for tag, G_ba, G_ab in (("raw", G_BA, G_AB), ("ema", ema_BA, ema_AB)):
            fake_b2a = metrics.jpeg_roundtrip(translate(G_ba, eval_photos, device, tc["precision"]), q)
            fake_a2b = metrics.jpeg_roundtrip(translate(G_ab, monets, device, tc["precision"]), q)
            s = metrics.competition_score(real_monet_f, real_photo_f, inception(fake_b2a), inception(fake_a2b))
            row.update({f"{k}_{tag}": v for k, v in s.items()})
            if s["score"] < best["score"]:
                best = {**s, "epoch": epoch, "iter": it, "weights": tag}
                half = lambda sd: {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}  # noqa: E731
                # stored in fp16 (~46 MB, fits GitHub); load_state_dict casts back to fp32
                torch.save({"G_AB": half(G_ab.state_dict()), "G_BA": half(G_ba.state_dict()), "model_cfg": mc, "weights": tag,
                            "epoch": epoch, "iter": it, "subset_score": s, "config": cfg}, best_path)
            G_BA.train()
            G_AB.train()
        evals.append(row)
        for tag in ("raw", "ema"):
            log.info(f"EVAL epoch {epoch} [{tag}] | FID photo->Monet {row['fid_B2A_' + tag]:.2f} "
                     f"Monet->photo {row['fid_A2B_' + tag]:.2f} avg {row['FID_' + tag]:.2f} | "
                     f"MiFID avg {row['MiFID_' + tag]:.4f} | score {row['score_' + tag]:.3f} "
                     f"(leaderboard ~ {-row['score_' + tag]:.2f})")
        log.info(f"best so far: score {best['score']:.3f} (epoch {best['epoch']}, {best['weights']} weights)")
        save_samples(ema_BA, ema_AB, sample_p, sample_m, device, tc["precision"], out_dir / "samples" / f"epoch_{epoch:03d}.jpg")

    # ---- training loop ----
    utils.reset_peak_memory(device)
    G_AB.train(), G_BA.train(), D_A.train(), D_B.train()
    acc, n_acc = {}, 0
    utils.sync(device)
    t_last = time.perf_counter()
    while it < total_iters:
        f = lr_factor(it)
        for g in opt_G.param_groups:
            g["lr"] = tc["lr_g"] * f
        for g in opt_D.param_groups:
            g["lr"] = tc["lr_d"] * f
        real_A = data.sample_batch(monets_dev, tc["batch_size"], tc["jitter"]).contiguous(memory_format=torch.channels_last)
        real_B = data.sample_batch(photos_dev, tc["batch_size"], tc["jitter"]).contiguous(memory_format=torch.channels_last)

        # -- generators --
        for p in params_D:
            p.requires_grad_(False)
        with autocast(device, tc["precision"]):
            fake_B, fake_A = G_AB(real_A), G_BA(real_B)
            rec_A, rec_B = G_BA(fake_B), G_AB(fake_A)
            pa, pb = D_A(aug(fake_A)), D_B(aug(fake_B))
            loss_gan = mse(pa.float(), torch.ones_like(pa, dtype=torch.float)) + \
                mse(pb.float(), torch.ones_like(pb, dtype=torch.float))
            loss_cyc = l1(rec_A.float(), real_A) + l1(rec_B.float(), real_B)
            if lam_i > 0:
                loss_idt = l1(G_BA(real_A).float(), real_A) + l1(G_AB(real_B).float(), real_B)
            else:
                loss_idt = torch.zeros((), device=device)
            loss_G = loss_gan + lam_c * loss_cyc + lam_i * loss_idt
        opt_G.zero_grad(set_to_none=True)
        loss_G.backward()
        gn_G = grad_norm(params_G)
        for p in params_D:
            p.requires_grad_(True)

        # -- discriminators --
        with autocast(device, tc["precision"]):
            fA, fB = pool_A.query(fake_A.float()), pool_B.query(fake_B.float())
            ra, fa_ = D_A(aug(real_A)), D_A(aug(fA))
            rb, fb_ = D_B(aug(real_B)), D_B(aug(fB))
            loss_DA = 0.5 * (mse(ra.float(), torch.ones_like(ra, dtype=torch.float)) +
                             mse(fa_.float(), torch.zeros_like(fa_, dtype=torch.float)))
            loss_DB = 0.5 * (mse(rb.float(), torch.ones_like(rb, dtype=torch.float)) +
                             mse(fb_.float(), torch.zeros_like(fb_, dtype=torch.float)))
        opt_D.zero_grad(set_to_none=True)
        (loss_DA + loss_DB).backward()
        gn_D = grad_norm(params_D)

        # Values stay on the GPU; one sync per iteration (the finite check), the rest at log time.
        vals = {"loss_G": loss_G, "loss_G_gan": loss_gan, "loss_cycle": loss_cyc, "loss_identity": loss_idt,
                "loss_D_A": loss_DA, "loss_D_B": loss_DB, "grad_norm_G": gn_G, "grad_norm_D": gn_D}
        vals = {k: v.detach().float() for k, v in vals.items()}
        if not torch.isfinite(torch.stack(list(vals.values()))).all().item():
            nan_steps += 1  # skip the update instead of corrupting the weights
            log.warning(f"iter {it}: non-finite loss/gradient - update skipped "
                        f"({ {k: round(v.item(), 4) for k, v in vals.items()} })")
            opt_G.zero_grad(set_to_none=True)
            opt_D.zero_grad(set_to_none=True)
        else:
            opt_G.step()
            opt_D.step()
            decay = min(tc["ema_decay"], (1 + it) / (10 + it))
            ema_update(ema_AB, G_AB, decay)
            ema_update(ema_BA, G_BA, decay)
            for k, v in vals.items():
                acc[k] = acc.get(k, 0.0) + v
            n_acc += 1
        it += 1

        if it % tc["log_every"] == 0 or it == total_iters:
            utils.sync(device)
            dt = time.perf_counter() - t_last
            train_secs += dt
            row = {"iter": it, "epoch": it / iters_per_epoch, "lr_factor": f,
                   **{k: (v / max(1, n_acc)).item() for k, v in acc.items()},
                   "images_per_s": tc["log_every"] * tc["batch_size"] * 2 / dt, "nan_steps": nan_steps}
            hist.append(row)
            log.info(f"it {it:6d}/{total_iters} | ep {row['epoch']:6.2f} | G {row.get('loss_G', 0):.3f} "
                     f"(gan {row.get('loss_G_gan', 0):.3f} cyc {row.get('loss_cycle', 0):.3f} idt {row.get('loss_identity', 0):.3f}) "
                     f"| D_A {row.get('loss_D_A', 0):.3f} D_B {row.get('loss_D_B', 0):.3f} | gnG {row.get('grad_norm_G', 0):.2f} "
                     f"gnD {row.get('grad_norm_D', 0):.2f} | lr x{f:.2f} | {row['images_per_s']:.0f} img/s")
            acc, n_acc = {}, 0
            t_last = time.perf_counter()

        epoch_done = it % iters_per_epoch == 0 or it == total_iters
        epoch = math.ceil(it / iters_per_epoch)
        if epoch_done and (epoch % tc["eval_every"] == 0 or it == total_iters):
            evaluate(epoch)
            t_last = time.perf_counter()
        if epoch_done and (epoch % tc["ckpt_every"] == 0 or it == total_iters):
            save_last()
            t_last = time.perf_counter()

    # ---- wrap up ----
    peak = utils.peak_memory_mb(device)
    info = {"run_name": run, "iterations": it, "epochs": tc["epochs"], "iters_per_epoch": iters_per_epoch,
            "batch_size": tc["batch_size"], "train_time_s": train_secs, "train_time_min": train_secs / 60,
            "train_images_per_s": it * tc["batch_size"] * 2 / max(train_secs, 1e-9), "nan_steps": nan_steps,
            "parameters": n_params, "parameters_total": sum(n_params.values()), "best": best, **peak,
            "device": env["device_name"]}
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    for name, rows in (("train_history.csv", hist), ("eval_history.csv", evals)):
        if rows:
            with open(out_dir / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader()
                w.writerows(rows)
    plot_history(hist, evals, out_dir)
    manifest = utils.write_manifest(rid, cfg, device, log_path, checkpoints=[best_path], metrics=info,
                                    extra={"kind": "train", "outputs": utils.rel(out_dir)})
    log.info(f"best: {best}")
    log.info(f"raw log  -> {utils.rel(log_path)}")
    log.info(f"manifest -> {utils.rel(manifest)}")


if __name__ == "__main__":
    main()
