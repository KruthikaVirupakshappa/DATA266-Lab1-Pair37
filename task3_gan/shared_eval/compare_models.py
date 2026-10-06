"""Compare both members' Task 3 CycleGANs with identical metric code and settings.

Works on the saved prediction folders (exactly what each member submitted):
    <member>/outputs/pred_B2A/  photo -> Monet (the first 300 photos by file name are scored)
    <member>/outputs/pred_A2B/  Monet -> photo (all 300 Monets)
Every metric uses the same 300 + 300 images per member and the same settings:
    FID / MiFID   exactly the instructor's evaluation procedure (first 300 files, official Inception
                  preprocessing); both directions + submission averages
    KID           100 subsets of 100 images, cubic polynomial kernel (mean, std)
    precision / recall / density / coverage   k = 5 nearest neighbours in Inception feature space
    content cosine   Inception features of each input vs its translation
    LPIPS (AlexNet)  input vs translation
    colour        mean saturation / contrast of photo->Monet outputs vs real Monets
Cycle-reconstruction metrics need each member's generators, so they are taken from each member's
own report (see the notes in comparison_table.md).

Sanity check: the recomputed submission FID / MiFID must match each member's submission.csv.

Usage (from the repo root):
    python task3_gan/shared_eval/compare_models.py [--member NAME=DIR ...]
    default members: kruthika=task3_gan/kruthika  jiayi=task3_gan/jiayi
Writes task3_gan/shared_eval/{comparison_report.csv, comparison_table.md} and a raw log in
reproducibility/raw_logs/.
"""
import argparse
import csv
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "task3_gan" / "kruthika" / "src"))
import data as kdata  # noqa: E402
import metrics as km  # noqa: E402

DATA = ROOT / "task3_gan" / "data"
N = km.N_EVAL  # 300, as in the evaluation script
KID_SUBSETS, KID_SIZE, PRDC_K = 100, 100, 5

NOTES = """## How to read this table

- **Identical code and data for both models:** every number in the main table is computed by this script from the
  saved prediction folders, on the same 300 + 300 images per member (the images the evaluation script scores).
- **Sanity check:** the recomputed submission FID / MiFID matches each member's `submission.csv` (small differences of
  about 0.05 FID come from different PyTorch/Pillow versions, which decode and resize images slightly differently).
- **Kaggle score** is each member's own submission: −(FID + MiFID) / 2 of their `submission.csv`. The team's
  leaderboard entry shows the best of the two.
- **Cycle-reconstruction L1** needs the generators, so it is quoted from each member's own report, on a common scale:
  mean absolute pixel difference on a 0–1 scale. A value computed on images in [−1, 1] is divided by 2.
- **Human audit:** each member's 30 samples were rated by a different pair of raters (Kruthika's: two friends;
  Jiayi's: two family members), so the human scores describe each model separately and are **not comparable
  across models**; agreement (κ) is comparable as a measure of how consistent each pair of raters was.
- **Colour:** real Monets have mean saturation 0.305 and contrast 0.180; values closer to these mean a more Monet-like
  palette.
"""


def colour_stats(images_uint8):
    """Mean HSV saturation and mean contrast (std of grey level) of a set of images."""
    x = images_uint8.float() / 255.0
    mx, mn = x.max(1).values, x.min(1).values
    sat = ((mx - mn) / mx.clamp(min=1e-6)).mean((1, 2))
    return {"saturation": sat.mean().item(), "contrast": x.mean(1).flatten(1).std(1).mean().item()}


CRITERIA = ("style", "content", "artifacts")


def load_audit(member_dir):
    """Two raters' 1-5 scores per sample with the translation direction, from either layout:
    - outputs/human_audit/ratings_rater1.csv + ratings_rater2.csv + audit_key_DO_NOT_SHOW_RATERS.csv
    - outputs/human_audit_ratings.csv (one row per sample and rater: sample_id, direction, rater_id, ...)
    Returns {"direction": [...], "<criterion>": (scores rater 1, scores rater 2)} or None."""
    h = member_dir / "outputs" / "human_audit"
    if (h / "ratings_rater1.csv").exists():
        key = {r["sample_id"]: r["direction"] for r in csv.DictReader(open(h / "audit_key_DO_NOT_SHOW_RATERS.csv"))}
        r1, r2 = (list(csv.DictReader(open(h / f"ratings_rater{i}.csv"))) for i in (1, 2))
        dirs = ["photo->monet" if key[r["sample_id"]] == "photo->monet" else "monet->photo" for r in r1]
    elif (member_dir / "outputs" / "human_audit_ratings.csv").exists():
        rows = list(csv.DictReader(open(member_dir / "outputs" / "human_audit_ratings.csv")))
        raters = sorted({r["rater_id"] for r in rows})
        ids = sorted({r["sample_id"] for r in rows})
        by = {(r["sample_id"], r["rater_id"]): r for r in rows}
        r1, r2 = ([by[(i, rt)] for i in ids] for rt in raters[:2])
        dirs = ["photo->monet" if "monet" in by[(i, raters[0])]["direction"].lower().split("_to_")[-1] else "monet->photo" for i in ids]
    else:
        return None
    out = {"direction": dirs}
    for c in CRITERIA:
        out[c] = (np.array([int(r[c]) for r in r1]), np.array([int(r[c]) for r in r2]))
    return out


def audit_stats(a):
    from sklearn.metrics import cohen_kappa_score
    res, d = {}, np.array(a["direction"])
    for c in CRITERIA:
        x, y = a[c]
        res[f"{c}_kappa"] = cohen_kappa_score(x, y, labels=[1, 2, 3, 4, 5])
        res[f"{c}_kappa_quadratic"] = cohen_kappa_score(x, y, labels=[1, 2, 3, 4, 5], weights="quadratic")
        res[f"{c}_exact_agreement"] = float((x == y).mean())
        for name in ("photo->monet", "monet->photo"):
            m = d == name
            res[f"{c}_mean_{name}"] = float(np.concatenate([x[m], y[m]]).mean())
    return res


def first_n(folder, n=N):
    return sorted(Path(folder).glob("*.jpg"))[:n]


def load_uint8(paths):
    from PIL import Image
    arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def read_submission(path):
    with open(path) as f:
        row = next(csv.DictReader(f))
    return float(row["FID"]), float(row["MiFID"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", help="NAME=DIR (DIR contains outputs/pred_A2B, outputs/pred_B2A, submission.csv)")
    args = ap.parse_args()
    members = dict(m.split("=", 1) for m in (args.member or ["kruthika=task3_gan/kruthika", "jiayi=task3_gan/jiayi"]))
    members = {k: (Path(v) if Path(v).is_absolute() else ROOT / v) for k, v in members.items()}

    log_dir = ROOT / "reproducibility" / "raw_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"task3_shared_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)])
    log = logging.info
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    log(f"python={platform.python_version()} torch={torch.__version__} device={device} members={list(members)}")

    inc = km.InceptionFeatures(device)
    lp = km.LPIPS(device)
    real_monet_p, real_photo_p = first_n(DATA / "monet_jpg"), first_n(DATA / "photo_jpg")
    real_monet_f, real_photo_f = inc.official_features([str(p) for p in real_monet_p]), inc.official_features([str(p) for p in real_photo_p])
    real_monet_u8 = load_uint8(real_monet_p)
    ref_colour = colour_stats(real_monet_u8)
    log(f"real Monet colour: {ref_colour}")

    results = {}
    for name, d in members.items():
        b2a_p, a2b_p = first_n(d / "outputs" / "pred_B2A"), first_n(d / "outputs" / "pred_A2B")
        assert [p.name for p in b2a_p] == [p.name for p in real_photo_p], f"{name}: pred_B2A names differ from the first 300 photos"
        assert sorted(p.name for p in a2b_p) == sorted(p.name for p in real_monet_p), f"{name}: pred_A2B names differ from the Monets"
        f_b2a, f_a2b = inc.official_features([str(p) for p in b2a_p]), inc.official_features([str(p) for p in a2b_p])
        s = km.competition_score(real_monet_f, real_photo_f, f_b2a, f_a2b)
        sub_fid, sub_mifid = read_submission(d / "submission.csv")
        log(f"SANITY {name}: recomputed FID {s['FID']:.4f} MiFID {s['MiFID']:.4f} vs submission.csv "
            f"{sub_fid:.4f} / {sub_mifid:.4f} (diff {s['FID'] - sub_fid:+.4f} / {s['MiFID'] - sub_mifid:+.4f})")
        assert abs(s["FID"] - sub_fid) < 0.5 and abs(s["MiFID"] - sub_mifid) < 0.01, f"{name}: does not match submission.csv"

        # inputs for content metrics: photo for B2A (same names), Monet for A2B (same names)
        in_b2a_p = [DATA / "photo_jpg" / p.name for p in b2a_p]
        in_a2b_p = [DATA / "monet_jpg" / p.name for p in a2b_p]
        f_in_b2a, f_in_a2b = inc.official_features([str(p) for p in in_b2a_p]), inc.official_features([str(p) for p in in_a2b_p])
        u_b2a, u_a2b, u_in_b2a, u_in_a2b = load_uint8(b2a_p), load_uint8(a2b_p), load_uint8(in_b2a_p), load_uint8(in_a2b_p)

        r = {"submission_FID": sub_fid, "submission_MiFID": sub_mifid, "kaggle_score": -(sub_fid + sub_mifid) / 2,
             "recomputed_FID": s["FID"], "recomputed_MiFID": s["MiFID"]}
        for tag, fg, fr, fin, ug, uin in (("B2A", f_b2a, real_monet_f, f_in_b2a, u_b2a, u_in_b2a),
                                          ("A2B", f_a2b, real_photo_f, f_in_a2b, u_a2b, u_in_a2b)):
            r[f"FID_{tag}"] = s[f"fid_{tag}"]
            r[f"MiFID_{tag}"] = s[f"mifid_{tag}"]
            r[f"KID_{tag}"], r[f"KID_std_{tag}"] = km.kid(fg, fr, n_subsets=KID_SUBSETS, subset_size=KID_SIZE)
            for k, v in km.prdc(fr, fg, k=PRDC_K).items():
                r[f"{k}_{tag}"] = v
            r[f"content_cosine_{tag}"] = km.content_cosine(fin, fg)
            r[f"LPIPS_input_vs_output_{tag}"] = lp(uin, ug)
        for k, v in colour_stats(u_b2a).items():
            r[f"B2A_{k}"] = v
        results[name] = r
        log(f"RESULT {name}: " + " | ".join(f"{k} {v:.4f}" for k, v in r.items()))

    # ---- outputs ----
    names = list(results)
    keys = list(results[names[0]])
    with open(HERE / "comparison_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", *names])
        for k in keys:
            w.writerow([k, *[results[n][k] for n in names]])
        w.writerow(["real_monet_saturation_reference", *[ref_colour["saturation"]] * len(names)])
        w.writerow(["real_monet_contrast_reference", *[ref_colour["contrast"]] * len(names)])

    def cells(fn):
        return " | ".join(fn(results[n]) for n in names)

    def row(label, key, d=4):
        return f"| {label} | " + cells(lambda r: f"{r[key]:.{d}f}") + " |"
    blank = " | " * (len(names) - 1)
    lines = ["| Metric | " + " | ".join(n.capitalize() for n in names) + " |", "|---|" + "---|" * len(names),
             row("**Kaggle score** (own submission)", "kaggle_score"),
             row("Submission FID (average of both directions)", "submission_FID", 2),
             row("Submission MiFID (average)", "submission_MiFID")]
    for tag, title in (("B2A", "*Photo → Monet (B2A)*"), ("A2B", "*Monet → photo (A2B)*")):
        lines += [f"| {title} |{blank} |",
                  row("FID", f"FID_{tag}", 2), row("MiFID", f"MiFID_{tag}"),
                  f"| KID (mean ± std) | " + cells(lambda r: f"{r['KID_' + tag]:.4f} ± {r['KID_std_' + tag]:.4f}") + " |",
                  f"| Precision / recall (k={PRDC_K}) | " + cells(lambda r: f"{r['precision_' + tag]:.3f} / {r['recall_' + tag]:.3f}") + " |",
                  f"| Density / coverage (k={PRDC_K}) | " + cells(lambda r: f"{r['density_' + tag]:.3f} / {r['coverage_' + tag]:.3f}") + " |",
                  row("Content cosine (input vs output)", f"content_cosine_{tag}", 3),
                  row("LPIPS (input vs output)", f"LPIPS_input_vs_output_{tag}", 3)]
        if tag == "B2A":
            lines.append(f"| Saturation / contrast (real Monets {ref_colour['saturation']:.3f} / {ref_colour['contrast']:.3f}) | "
                         + cells(lambda r: f"{r['B2A_saturation']:.3f} / {r['B2A_contrast']:.3f}") + " |")
    audits = {n: load_audit(d) for n, d in members.items()}
    audit_res = {n: audit_stats(a) for n, a in audits.items() if a is not None}
    if len(audit_res) == len(names):
        for n in names:
            log(f"AUDIT {n}: " + " | ".join(f"{k} {v:.3f}" for k, v in audit_res[n].items()))
        lines += ["", "### Human audit (computed with identical code; **different raters per model**)", "",
                  "| Criterion | " + " | ".join(n.capitalize() for n in names) + " |", "|---|" + "---|" * len(names)]
        for c in CRITERIA:
            lines.append(f"| {c.capitalize()}: mean photo→Monet / Monet→photo | " + " | ".join(
                f"{audit_res[n][c + '_mean_photo->monet']:.2f} / {audit_res[n][c + '_mean_monet->photo']:.2f}" for n in names) + " |")
        for c in CRITERIA:
            lines.append(f"| {c.capitalize()}: Cohen's κ / quadratic-weighted κ | " + " | ".join(
                f"{audit_res[n][c + '_kappa']:.3f} / {audit_res[n][c + '_kappa_quadratic']:.3f}" for n in names) + " |")
        with open(HERE / "comparison_report.csv", "a", newline="") as f:
            w = csv.writer(f)
            for k in audit_res[names[0]]:
                w.writerow([f"audit_{k}", *[audit_res[n][k] for n in names]])
    (HERE / "comparison_table.md").write_text(
        "# Task 3 model comparison (identical metric code)\n\n"
        f"Each member's submitted predictions: the first {N} photos → Monet and all {N} Monets → photo. "
        f"KID: {KID_SUBSETS} subsets of {KID_SIZE}; PRDC: k = {PRDC_K}.\n\n" + "\n".join(lines) + "\n\n" + NOTES)
    log(f"outputs -> {(HERE / 'comparison_report.csv').relative_to(ROOT)}, comparison_table.md")
    log(f"raw log -> {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
