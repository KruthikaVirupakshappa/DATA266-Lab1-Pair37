"""Compare all six Task 2 models (3 per member) with identical metric code and settings.

Both members evaluated on the official Yelp Polarity test split (38,000 reviews) and saved the
predicted probability of the positive class for every review in the official order:
    task2_sentiment/kruthika/outputs/test_predictions.csv
    task2_sentiment/jiayi/outputs/test_predictions.csv
This script recomputes every metric of the brief for all six models with Kruthika's evaluation
code (task2_sentiment/kruthika/src/evaluate.py), so the settings are guaranteed to be identical:
    threshold 0.5; accuracy; precision/recall/F1 (macro, micro, weighted); confusion matrix;
    ROC-AUC; PR-AUC (average precision); MCC; Brier score; ECE (15 bins, confidence of the
    predicted class); 95% bootstrap CIs (1,000 resamples, seed 42) for accuracy, macro-F1 and MCC;
    paired McNemar tests (each member's baseline vs their experiments, and across members);
    macro-F1 and error rate per slice (review length, negation, contrast - same slices for all).
Efficiency numbers (parameters, training time, examples/s, memory) cannot be recomputed from
predictions and are taken from each member's own report.

Sanity checks: identical test labels in both files, and recomputed accuracies equal to each
member's reported accuracies.

Usage (from the repo root):
    python task2_sentiment/shared_eval/compare_models.py [--jiayi-dir task2_sentiment/jiayi]
Writes task2_sentiment/shared_eval/{comparison_report.csv, comparison_table.md, mcnemar.csv,
slice_metrics.csv} and a raw log in reproducibility/raw_logs/.
"""
import argparse
import csv
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "task2_sentiment" / "kruthika" / "src"))
import evaluate as ev  # noqa: E402

K_DIR = ROOT / "task2_sentiment" / "kruthika"
SLICES = ("length", "negation", "contrast")
SEED, N_BOOT, ECE_BINS = 42, 1000, 15

NOTES = """## How to read this table

- **Identical code and data:** every quality metric above is recomputed by this script from the two members'
  `test_predictions.csv` files (official test split, 38,000 reviews, same order and labels) with the same settings.
  The sanity check confirms that the recomputed accuracies equal each member's reported accuracies exactly.
- **Different settings in the members' own reports:** Jiayi's notebook computed ECE with 10 bins over P(positive)
  and PR-AUC as the trapezoidal area under the precision-recall curve, and used different length slices. Kruthika's
  computed ECE with 15 bins over the confidence of the predicted class and PR-AUC as average precision. This table
  uses one definition for all six models, so ECE, PR-AUC and slice values here can differ slightly from the
  individual reports.
- **Slices** use the same definitions for all models: review length in raw words (≤50 / 51–200 / >200),
  whether the review contains a negation, and whether it contains a contrast word (but / however / although /
  though / yet).
- **Efficiency rows** come from each member's own training run (Google Colab GPU; exact GPU in each member's
  metrics report). They are not measured identically: Kruthika's runs use batch size 256 and keep the whole
  encoded dataset on the GPU (about 0.6 GB, which dominates her peak memory); Jiayi's use batch size 64 with a
  data loader. Compare them as orders of magnitude.
- **McNemar** uses the chi-squared test with continuity correction on the same 38,000 reviews; p < 0.05 means the
  two models' error patterns differ significantly.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jiayi-dir", default="task2_sentiment/jiayi")
    args = ap.parse_args()
    J_DIR = Path(args.jiayi_dir) if Path(args.jiayi_dir).is_absolute() else ROOT / args.jiayi_dir

    log_dir = ROOT / "reproducibility" / "raw_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"task2_shared_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)])
    log = logging.info
    log(f"python={platform.python_version()}")

    kp = pd.read_csv(K_DIR / "outputs" / "test_predictions.csv")
    jp = pd.read_csv(J_DIR / "outputs" / "test_predictions.csv")
    assert len(kp) == len(jp) == 38000, (len(kp), len(jp))
    assert (kp["test_index"].values == jp["test_index"].values).all(), "test_index order differs"
    assert (kp["true"].values == jp["true"].values).all(), "true labels differ between the two files"
    y = kp["true"].values.astype(int)
    slices = {s: kp[s].values for s in SLICES}
    log(f"sanity: both files have the same 38,000 test reviews in the same order with identical labels")

    # model name -> (member, column, short description)
    kcfg = yaml.safe_load(open(K_DIR / "src" / "config.yaml"))["models"]
    jcfg = json.loads((J_DIR / "config.json").read_text())
    models = {
        "K-baseline (mean embedding)": ("kruthika", "prob_baseline_mean_embedding",
            f"mean of word embeddings (d={kcfg['baseline_mean_embedding']['embed_dim']}) → linear; dropout "
            f"{kcfg['baseline_mean_embedding']['dropout']}; Adam lr {kcfg['baseline_mean_embedding']['lr']}"),
        "K-exp1 (BiLSTM)": ("kruthika", "prob_exp1_bilstm",
            f"embeddings (d={kcfg['exp1_bilstm']['embed_dim']}) → 1-layer BiLSTM ({kcfg['exp1_bilstm']['hidden']}/direction) "
            f"→ max+mean pooling → linear; dropout {kcfg['exp1_bilstm']['dropout']}; lr {kcfg['exp1_bilstm']['lr']}"),
        "K-exp2 (TextCNN)": ("kruthika", "prob_exp2_textcnn",
            f"embeddings (d={kcfg['exp2_textcnn']['embed_dim']}) → Conv1d {kcfg['exp2_textcnn']['kernel_sizes']} × "
            f"{kcfg['exp2_textcnn']['filters']} filters → max-over-time → linear; dropout {kcfg['exp2_textcnn']['dropout']}; "
            f"lr {kcfg['exp2_textcnn']['lr']}"),
        "J-baseline (mean+max embedding)": ("jiayi", "prob_meanmax",
            f"[mean ; max] of word embeddings (d={jcfg['baseline']['embedding_dim']}) → linear; lr {jcfg['baseline']['learning_rate']}"),
        "J-exp1 (MLP)": ("jiayi", "prob_mlp",
            f"mean of word embeddings (d={jcfg['experiment_1']['embedding_dim']}) → MLP {jcfg['experiment_1']['hidden_dim_1']}→"
            f"{jcfg['experiment_1']['hidden_dim_2']}→1; dropout {jcfg['experiment_1']['dropout']}; lr {jcfg['experiment_1']['learning_rate']}"),
        "J-exp2 (self-attention)": ("jiayi", "prob_attention",
            f"word + position embeddings (d={jcfg['experiment_2']['embedding_dim']}) → single-head self-attention "
            f"(d={jcfg['experiment_2']['attention_dim']}) → mean pooling → linear; lr {jcfg['experiment_2']['learning_rate']}"),
    }
    preds = {m: (kp if who == "kruthika" else jp)[col].values.astype(float) for m, (who, col, _) in models.items()}

    # ---- sanity: recomputed accuracy == reported accuracy ----
    k_rep = pd.read_csv(K_DIR / "outputs" / "comparison.csv").set_index("model")["accuracy"]
    j_rep = pd.read_csv(J_DIR / "metrics_report.csv").set_index("model")["accuracy"]
    reported = {"K-baseline (mean embedding)": k_rep["baseline_mean_embedding"], "K-exp1 (BiLSTM)": k_rep["exp1_bilstm"],
                "K-exp2 (TextCNN)": k_rep["exp2_textcnn"], "J-baseline (mean+max embedding)": j_rep["baseline"],
                "J-exp1 (MLP)": j_rep["experiment_1_mlp"], "J-exp2 (self-attention)": j_rep["experiment_2_attention"]}
    for m, p in preds.items():
        acc = float(((p >= 0.5).astype(int) == y).mean())
        log(f"SANITY {m}: recomputed accuracy {acc:.6f} vs reported {reported[m]:.6f}")
        assert abs(acc - reported[m]) < 1e-6, f"{m}: accuracy does not match the member's report"

    # ---- metrics ----
    res = {}
    for m, p in preds.items():
        r = ev.classification_metrics(y, p, ECE_BINS)
        r.update(ev.bootstrap_ci(y, p, N_BOOT, seed=SEED))
        res[m] = r
        log(f"{m}: acc {r['accuracy']:.4f} [{r['accuracy_ci_low']:.4f}, {r['accuracy_ci_high']:.4f}] macroF1 {r['f1_macro']:.4f} "
            f"MCC {r['mcc']:.4f} ROC-AUC {r['roc_auc']:.4f} PR-AUC {r['pr_auc']:.4f} Brier {r['brier']:.4f} ECE {r['ece']:.4f}")

    # efficiency from each member's own report
    k_eff = pd.read_csv(K_DIR / "outputs" / "comparison.csv").set_index("model")
    j_eff = pd.read_csv(J_DIR / "metrics_report.csv").set_index("model")
    eff_map = {"K-baseline (mean embedding)": ("k", "baseline_mean_embedding"), "K-exp1 (BiLSTM)": ("k", "exp1_bilstm"),
               "K-exp2 (TextCNN)": ("k", "exp2_textcnn"), "J-baseline (mean+max embedding)": ("j", "baseline"),
               "J-exp1 (MLP)": ("j", "experiment_1_mlp"), "J-exp2 (self-attention)": ("j", "experiment_2_attention")}
    for m, (who, key) in eff_map.items():
        if who == "k":
            e = k_eff.loc[key]
            res[m].update({"parameters": e["parameters"], "train_time_s": e["train_time_s"],
                           "train_examples_per_s": e["train_examples_per_s"], "inference_examples_per_s": e["inference_examples_per_s"],
                           "peak_gpu_mem_mb": e["peak_gpu_mem_mb"]})
        else:
            e = j_eff.loc[key]
            res[m].update({"parameters": e["parameter_count"], "train_time_s": e["training_time_sec"],
                           "train_examples_per_s": e["training_examples_per_sec"], "inference_examples_per_s": float("nan"),
                           "peak_gpu_mem_mb": e["peak_gpu_memory_mb"]})

    # ---- slices ----
    slice_rows = []
    for m, p in preds.items():
        for r in ev.slice_metrics(y, p, slices):
            slice_rows.append({"model": m, **r})

    # ---- McNemar ----
    pairs = [("K-baseline (mean embedding)", "K-exp1 (BiLSTM)"), ("K-baseline (mean embedding)", "K-exp2 (TextCNN)"),
             ("J-baseline (mean+max embedding)", "J-exp1 (MLP)"), ("J-baseline (mean+max embedding)", "J-exp2 (self-attention)")]
    k_best = max([m for m in models if m.startswith("K")], key=lambda m: res[m]["accuracy"])
    j_best = max([m for m in models if m.startswith("J")], key=lambda m: res[m]["accuracy"])
    pairs += [("J-baseline (mean+max embedding)", "K-baseline (mean embedding)"), (j_best, k_best)]
    mcn = []
    for a, b in pairs:
        t = ev.mcnemar(y, preds[a], preds[b])
        mcn.append({"model_A": a, "model_B": b, "A_only_correct": t["b_base_only_correct"], "B_only_correct": t["c_exp_only_correct"],
                    "chi2": t["chi2"], "p_value": t["p_value"], "p_value_exact": t["p_value_exact"]})
        log(f"McNemar {a} vs {b}: A-only {t['b_base_only_correct']} B-only {t['c_exp_only_correct']} "
            f"chi2 {t['chi2']:.2f} p {t['p_value']:.3g}")

    # ---- write ----
    with open(HERE / "comparison_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "member", "metric", "value"])
        for m, r in res.items():
            for k, v in r.items():
                w.writerow([m, models[m][0], k, v])
    pd.DataFrame(slice_rows).to_csv(HERE / "slice_metrics.csv", index=False)
    pd.DataFrame(mcn).to_csv(HERE / "mcnemar.csv", index=False)

    names = list(models)
    head = "| | " + " | ".join(names) + " |"
    sep = "|---|" + "---|" * len(names)

    def row(label, fn):
        return f"| {label} | " + " | ".join(fn(m) for m in names) + " |"
    f4 = lambda k: (lambda m: f"{res[m][k]:.4f}")  # noqa: E731
    ci = lambda k: (lambda m: f"{res[m][k]:.4f} [{res[m][k + '_ci_low']:.4f}, {res[m][k + '_ci_high']:.4f}]")  # noqa: E731
    num = lambda k, d=0: (lambda m: "not reported" if pd.isna(res[m][k]) else f"{float(res[m][k]):,.{d}f}")  # noqa: E731
    table = [head, sep,
             row("Member", lambda m: models[m][0].capitalize()),
             row("Architecture / key hyperparameters", lambda m: models[m][2]),
             row("**Accuracy** [95% CI]", ci("accuracy")),
             row("Precision macro / micro / weighted", lambda m: f"{res[m]['precision_macro']:.4f} / {res[m]['precision_micro']:.4f} / {res[m]['precision_weighted']:.4f}"),
             row("Recall macro / micro / weighted", lambda m: f"{res[m]['recall_macro']:.4f} / {res[m]['recall_micro']:.4f} / {res[m]['recall_weighted']:.4f}"),
             row("F1 micro / weighted", lambda m: f"{res[m]['f1_micro']:.4f} / {res[m]['f1_weighted']:.4f}"),
             row("**Macro-F1** [95% CI]", ci("f1_macro")),
             row("Confusion matrix TN / FP / FN / TP", lambda m: f"{res[m]['tn']:,} / {res[m]['fp']:,} / {res[m]['fn']:,} / {res[m]['tp']:,}"),
             row("ROC-AUC", f4("roc_auc")), row("PR-AUC (average precision)", f4("pr_auc")),
             row("**MCC** [95% CI]", ci("mcc")), row("Brier score", f4("brier")), row(f"ECE ({ECE_BINS} bins)", f4("ece"))]
    for s in SLICES:
        for v in sorted(set(slices[s])):
            sub = {r["model"]: r for r in slice_rows if r["slice"] == s and r["value"] == v}
            n = next(iter(sub.values()))["n"]
            table.append(row(f"Slice {v} (n={n:,}): macro-F1 / error rate",
                             lambda m: f"{sub[m]['f1_macro']:.4f} / {sub[m]['error_rate']:.2%}"))
    table += [row("Parameters", num("parameters")), row("Training time (s)", num("train_time_s")),
              row("Training examples/s", num("train_examples_per_s")), row("Inference examples/s", num("inference_examples_per_s")),
              row("Peak GPU memory (MB)", num("peak_gpu_mem_mb", 1))]
    mtab = ["| Model A | Model B | A right, B wrong | A wrong, B right | χ² | p-value |", "|---|---|---|---|---|---|"]
    mtab += [f"| {r['model_A']} | {r['model_B']} | {r['A_only_correct']:,} | {r['B_only_correct']:,} | {r['chi2']:.2f} | {r['p_value']:.3g} |" for r in mcn]
    (HERE / "comparison_table.md").write_text(
        "# Task 2 model comparison (all six models, identical metric code)\n\n"
        "Official Yelp Polarity test split, 38,000 reviews; threshold 0.5. K = Kruthika, J = Jiayi. All embeddings "
        "are learned from scratch.\n\n" + "\n".join(table) +
        "\n\n### Paired McNemar tests\n\n" + "\n".join(mtab) + "\n\n" + NOTES)
    log(f"outputs -> {(HERE / 'comparison_table.md').relative_to(ROOT)}, comparison_report.csv, mcnemar.csv, slice_metrics.csv")
    log(f"raw log -> {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
