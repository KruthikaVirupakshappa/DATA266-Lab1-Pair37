"""Task 3.2 - generate predictions with the trained CycleGAN and evaluate them.

Usage (from the repo root):
    python task3_gan/kruthika/evaluate_local.py              # best checkpoint of the run in src/config.yaml
    python task3_gan/kruthika/evaluate_local.py --audit      # only compute human-audit agreement from ratings

Writes (in task3_gan/kruthika/):
    outputs/pred_A2B/          Monet -> photo, all 300 Monets (same file names)
    outputs/pred_B2A/          photo -> Monet, all 7,038 photos (same file names) - the scored set
    submission.csv             Kaggle file: ID, FID, MiFID - computed exactly as the instructor's
                               evaluation script (Part3_Evaluation_Script.ipynb): first 300 files of
                               each folder, both directions, average FID and average MiFID
    full_metrics_report.csv    every Task 3 metric, both directions
    outputs/human_audit/       30 fixed, anonymised samples + rating sheets for 2 raters
Scores are computed from the saved JPG files, i.e. exactly what is submitted.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MEMBER_DIR = Path(__file__).resolve().parent
SRC = MEMBER_DIR / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(MEMBER_DIR.parent / "scripts"))

import data  # noqa: E402
import metrics  # noqa: E402
import utils  # noqa: E402
from download_kaggle_data import download_kaggle_data  # noqa: E402
from models import Generator  # noqa: E402
from train import autocast, translate  # noqa: E402

AUDIT_DIR = MEMBER_DIR / "outputs" / "human_audit"
CRITERIA = ("style", "content", "artifacts")


def save_jpgs(images_uint8, names, folder, quality):
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.jpg"):
        old.unlink()
    for img, name in zip(images_uint8, names):
        Image.fromarray(img.permute(1, 2, 0).numpy()).save(folder / name, quality=quality)


def human_audit_sheet(photos, monets, fake_photo2monet, fake_monet2photo, photo_names, monet_names, n, seed):
    """30 fixed samples (half per direction), shuffled under anonymous IDs. Each image shows
    input | output side by side; raters do not see the direction key or each other's ratings."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    half = n // 2
    items = [("photo->monet", i) for i in rng.choice(len(photos), half, replace=False)] + \
            [("monet->photo", i) for i in rng.choice(len(monets), n - half, replace=False)]
    order = rng.permutation(len(items))
    key = []
    for sid, k in enumerate(order, start=1):
        direction, i = items[k]
        src, out, name = (photos[i], fake_photo2monet[i], photo_names[i]) if direction == "photo->monet" \
            else (monets[i], fake_monet2photo[i], monet_names[i])
        pair = torch.cat([src, torch.full((3, 256, 8), 255, dtype=torch.uint8), out], dim=2)
        Image.fromarray(pair.permute(1, 2, 0).numpy()).save(AUDIT_DIR / f"sample_{sid:02d}.jpg", quality=95)
        key.append({"sample_id": sid, "direction": direction, "source_file": name})
    with open(AUDIT_DIR / "audit_key_DO_NOT_SHOW_RATERS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    for r in (1, 2):
        path = AUDIT_DIR / f"ratings_rater{r}.csv"
        if not path.exists():  # never overwrite ratings that were already filled in
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sample_id", *CRITERIA])
                w.writerows([[k["sample_id"], "", "", ""] for k in key])
    (AUDIT_DIR / "README.md").write_text(
        "# Blinded human audit (30 fixed samples)\n\n"
        "Each `sample_XX.jpg` shows the **input (left)** and the **model output (right)**. "
        "The translation direction is hidden; do not open `audit_key_DO_NOT_SHOW_RATERS.csv`.\n\n"
        "Each rater fills in their own `ratings_raterN.csv` independently, scoring every sample 1-5:\n"
        "- **style**: does the output look like the target domain (a Monet painting or a real photo)? 1 = not at all, 5 = convincing\n"
        "- **content**: is the input's scene/structure preserved? 1 = lost, 5 = fully preserved\n"
        "- **artifacts**: visual defects (blotches, checkerboard, colour bleeding)? 1 = severe, 5 = none\n\n"
        "Then run `python task3_gan/kruthika/evaluate_local.py --audit` to compute agreement.\n")


def audit_agreement():
    """Cohen's kappa (plain and quadratic-weighted), exact and within-1 agreement per criterion,
    mean scores per rater, and mean scores per translation direction (both raters averaged)."""
    from sklearn.metrics import cohen_kappa_score
    paths = [AUDIT_DIR / f"ratings_rater{r}.csv" for r in (1, 2)]
    rows = [list(csv.DictReader(open(p))) for p in paths]
    key = {r["sample_id"]: r["direction"] for r in csv.DictReader(open(AUDIT_DIR / "audit_key_DO_NOT_SHOW_RATERS.csv"))}
    out = {}
    for c in CRITERIA:
        try:
            a = np.array([int(r[c]) for r in rows[0]])
            b = np.array([int(r[c]) for r in rows[1]])
        except (ValueError, KeyError):
            print(f"ratings for '{c}' are incomplete - fill in both rating sheets first")
            return None
        out[f"audit_{c}_kappa"] = metrics.cohen_kappa(a, b)
        out[f"audit_{c}_kappa_quadratic"] = float(cohen_kappa_score(a, b, labels=[1, 2, 3, 4, 5], weights="quadratic"))
        out[f"audit_{c}_agreement"] = float((a == b).mean())
        out[f"audit_{c}_agreement_within1"] = float((np.abs(a - b) <= 1).mean())
        out[f"audit_{c}_mean_rater1"], out[f"audit_{c}_mean_rater2"] = float(a.mean()), float(b.mean())
        for d in sorted(set(key.values())):
            vals = [int(r[c]) for rr in rows for r in rr if key[r["sample_id"]] == d]
            out[f"audit_{c}_mean_{d.replace('->', '_to_')}"] = float(np.mean(vals))
    (AUDIT_DIR / "audit_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task3_gan/kruthika/src/config.yaml")
    ap.add_argument("--checkpoint", help="default: checkpoints/<run_name>_best.pt")
    ap.add_argument("--audit", action="store_true", help="only compute human-audit agreement")
    args = ap.parse_args()
    if args.audit:
        audit_agreement()
        return

    cfg = utils.load_config(args.config)
    ec, run = cfg["eval"], cfg["run_name"]
    precision = cfg["train"]["precision"]
    rid = utils.run_id(f"{run}_eval")
    log, log_path = utils.get_logger(rid)
    device = utils.pick_device(cfg.get("device", "auto"))
    env = utils.environment_info(device)
    log.info(f"run_id={rid} device={device} ({env['device_name']})")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else MEMBER_DIR / "checkpoints" / f"{run}_best.pt"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    G_AB, G_BA = Generator(**ck["model_cfg"]).to(device), Generator(**ck["model_cfg"]).to(device)
    G_AB.load_state_dict(ck["G_AB"])
    G_BA.load_state_dict(ck["G_BA"])
    log.info(f"checkpoint {utils.rel(ckpt_path)}: epoch {ck['epoch']}, {ck['weights']} weights, "
             f"training-subset score {ck['subset_score']['score']:.3f}")

    data_dir = utils.repo_path(cfg["data"]["data_dir"])
    download_kaggle_data()
    monets, monet_names = data.load_images(data_dir / "monet_jpg")
    photos, photo_names = data.load_images(data_dir / "photo_jpg")

    # ---- generate and save predictions ----
    utils.sync(device)
    t0 = time.perf_counter()
    fake_monet = translate(G_BA, photos, device, precision)     # photo -> Monet (B2A)
    fake_photo = translate(G_AB, monets, device, precision)     # Monet -> photo (A2B)
    utils.sync(device)
    gen_secs = time.perf_counter() - t0
    out = MEMBER_DIR / "outputs"
    save_jpgs(fake_monet, photo_names, out / "pred_B2A", ec["jpg_quality"])
    save_jpgs(fake_photo, monet_names, out / "pred_A2B", ec["jpg_quality"])
    log.info(f"generated {len(fake_monet)} B2A + {len(fake_photo)} A2B images in {gen_secs:.1f}s "
             f"({(len(fake_monet) + len(fake_photo)) / gen_secs:.0f} images/s)")

    inception = metrics.InceptionFeatures(device)

    # ---- Kaggle submission: exactly the evaluation script's procedure, on the files on disk ----
    # sorted file lists, first N_EVAL = 300 of each folder, official Inception preprocessing
    first = lambda folder: [str(p) for p in data.list_images(folder)][:metrics.N_EVAL]  # noqa: E731
    of = {k: inception.official_features(first(d)) for k, d in (
        ("real_monet", data_dir / "monet_jpg"), ("real_photo", data_dir / "photo_jpg"),
        ("gen_b2a", out / "pred_B2A"), ("gen_a2b", out / "pred_A2B"))}
    kag = metrics.competition_score(of["real_monet"], of["real_photo"], of["gen_b2a"], of["gen_a2b"])
    import pandas as pd
    pd.DataFrame([{"ID": 1, "FID": float(kag["FID"]), "MiFID": float(kag["MiFID"])}]).to_csv(
        MEMBER_DIR / "submission.csv", index=False)
    log.info(f"[Photo->Monet] FID={kag['fid_B2A']:.3f}  MiFID={kag['mifid_B2A']:.4f}")
    log.info(f"[Monet->Photo] FID={kag['fid_A2B']:.3f}  MiFID={kag['mifid_A2B']:.4f}")
    log.info(f"SUBMISSION FID={kag['FID']:.4f} MiFID={kag['MiFID']:.4f} | estimated leaderboard score "
             f"{-kag['score']:.4f}")

    # Features of the complete sets for the additional metrics (fast path)
    sub_B2A, _ = data.load_images(out / "pred_B2A")
    sub_A2B, _ = data.load_images(out / "pred_A2B")
    f_photo, f_monet = inception(photos), inception(monets)
    f_B2A, f_A2B = inception(sub_B2A), inception(sub_A2B)

    # ---- full metrics, both directions ----
    rows = []

    def add(direction, metric, value):
        rows.append({"direction": direction, "metric": metric, "value": value})
        log.info(f"METRIC [{direction}] {metric} = {value}")

    add("photo->monet (B2A)", "fid_eval_script", kag["fid_B2A"])
    add("photo->monet (B2A)", "mifid_eval_script", kag["mifid_B2A"])
    add("monet->photo (A2B)", "fid_eval_script", kag["fid_A2B"])
    add("monet->photo (A2B)", "mifid_eval_script", kag["mifid_A2B"])
    add("submission", "FID_average", kag["FID"])
    add("submission", "MiFID_average", kag["MiFID"])
    add("submission", "estimated_leaderboard_score", -kag["score"])
    # FID on the complete sets (all 7,038 translated photos vs 300 Monets; 300 translated Monets vs 7,038 photos)
    add("photo->monet (B2A)", "fid_all_images", metrics.fid(*metrics.gaussian_stats(f_B2A), *metrics.gaussian_stats(f_monet)))
    add("monet->photo (A2B)", "fid_all_images", metrics.fid(*metrics.gaussian_stats(f_A2B), *metrics.gaussian_stats(f_photo)))
    for d, fg, fr in (("photo->monet (B2A)", f_B2A, f_monet), ("monet->photo (A2B)", f_A2B, f_photo)):
        kmean, kstd = metrics.kid(fg, fr, ec["kid_subsets"])
        add(d, "kid_mean", kmean)
        add(d, "kid_std", kstd)
        for k, v in metrics.prdc(fr, fg, k=ec["prdc_k"]).items():
            add(d, k, v)
    add("photo->monet (B2A)", "content_cosine_input_vs_output", metrics.content_cosine(f_photo, f_B2A))
    add("monet->photo (A2B)", "content_cosine_input_vs_output", metrics.content_cosine(f_monet, f_A2B))

    # Cycle reconstruction, identity and LPIPS on a fixed photo subset and all Monets
    rng = np.random.default_rng(cfg["seed"])
    p_idx = np.sort(rng.choice(len(photos), min(ec["lpips_photos"], len(photos)), replace=False))
    lp = metrics.LPIPS(device)
    rec_photo = translate(G_AB, fake_monet[p_idx], device, precision)      # photo -> Monet -> photo
    rec_monet = translate(G_BA, fake_photo, device, precision)             # Monet -> photo -> Monet
    idt_photo = translate(G_AB, photos[p_idx], device, precision)          # G_AB(photo) should = photo
    idt_monet = translate(G_BA, monets, device, precision)                 # G_BA(Monet) should = Monet
    l1 = lambda a, b: float((a.float() - b.float()).abs().mean() / 255.0)  # noqa: E731  (pixel scale 0-1)
    add("photo->monet (B2A)", "cycle_l1_photo_monet_photo", l1(rec_photo, photos[p_idx]))
    add("monet->photo (A2B)", "cycle_l1_monet_photo_monet", l1(rec_monet, monets))
    add("photo->monet (B2A)", "identity_l1_G_AB_on_photo", l1(idt_photo, photos[p_idx]))
    add("monet->photo (A2B)", "identity_l1_G_BA_on_monet", l1(idt_monet, monets))
    add("photo->monet (B2A)", "lpips_input_vs_output", lp(photos[p_idx], fake_monet[p_idx]))
    add("monet->photo (A2B)", "lpips_input_vs_output", lp(monets, fake_photo))
    add("photo->monet (B2A)", "lpips_cycle_reconstruction", lp(photos[p_idx], rec_photo))
    add("monet->photo (A2B)", "lpips_cycle_reconstruction", lp(monets, rec_monet))

    # Training statistics
    info_path = MEMBER_DIR / "outputs" / run / "train_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        for k in ("parameters_total", "train_time_min", "train_images_per_s", "nan_steps", "iterations", "epochs",
                  "peak_gpu_mem_mb", "device"):
            if k in info:
                add("training", k, info[k])
        for k, v in info["parameters"].items():
            add("training", f"parameters_{k}", v)
        hist = list(csv.DictReader(open(MEMBER_DIR / "outputs" / run / "train_history.csv")))
        for k in ("grad_norm_G", "grad_norm_D"):
            v = [float(h[k]) for h in hist if h.get(k)]
            add("training", f"{k}_mean", float(np.mean(v)))
            add("training", f"{k}_max", float(np.max(v)))
        last = hist[-1]
        for k in ("loss_G_gan", "loss_cycle", "loss_identity", "loss_D_A", "loss_D_B"):
            add("training", f"final_{k}", float(last[k]))
    add("inference", "images_per_s", (len(fake_monet) + len(fake_photo)) / gen_secs)
    add("inference", "checkpoint", utils.rel(ckpt_path))
    add("inference", "checkpoint_epoch", ck["epoch"])
    add("inference", "checkpoint_weights", ck["weights"])

    human_audit_sheet(photos, monets, fake_monet, fake_photo, photo_names, monet_names, ec["audit_samples"], cfg["seed"])
    audit_file = AUDIT_DIR / "audit_results.json"
    if audit_file.exists():
        for k, v in json.loads(audit_file.read_text()).items():
            add("human audit", k, v)

    with open(MEMBER_DIR / "full_metrics_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "metric", "value"])
        w.writeheader()
        w.writerows(rows)
    manifest = utils.write_manifest(rid, cfg, device, log_path, checkpoints=[ckpt_path],
                                    metrics={r["direction"] + " | " + r["metric"]: r["value"] for r in rows},
                                    extra={"kind": "evaluate", "submission": {"FID": kag["FID"], "MiFID": kag["MiFID"]}})
    log.info(f"submission -> {utils.rel(MEMBER_DIR / 'submission.csv')}")
    log.info(f"metrics    -> {utils.rel(MEMBER_DIR / 'full_metrics_report.csv')}")
    log.info(f"manifest   -> {utils.rel(manifest)}")


if __name__ == "__main__":
    main()
