"""Task 2 evaluation metrics and plots (binary classification, positive class = 1).

All metrics take y (true labels 0/1) and p (predicted probability of class 1);
the predicted label is p >= 0.5.
"""
import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)


def expected_calibration_error(y, p, n_bins=15):
    """ECE: split predictions into confidence bins; weighted mean |accuracy - confidence|.
    Confidence = probability of the predicted class."""
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inb = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if inb.any():
            acc, cf = correct[inb].mean(), conf[inb].mean()
            ece += inb.mean() * abs(acc - cf)
            bins.append((lo, hi, int(inb.sum()), acc, cf))
    return ece, bins


def classification_metrics(y, p, ece_bins=15):
    pred = (p >= 0.5).astype(int)
    out = {"accuracy": accuracy_score(y, pred)}
    for avg in ("macro", "micro", "weighted"):
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = pr, rc, f1
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    out["roc_auc"] = roc_auc_score(y, p)
    out["pr_auc"] = average_precision_score(y, p)
    out["mcc"] = matthews_corrcoef(y, pred)
    out["brier"] = brier_score_loss(y, p)
    out["ece"] = expected_calibration_error(y, p, ece_bins)[0]
    out["error_rate"] = 1 - out["accuracy"]
    return out


def _counts_metrics(tp, fp, fn, tn):
    """Accuracy, macro-F1 and MCC from confusion counts (vectorised over bootstrap samples)."""
    n = tp + fp + fn + tn
    acc = (tp + tn) / n
    f1_pos = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1_neg = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.maximum(denom, 1e-12), 0.0)
    return acc, (f1_pos + f1_neg) / 2, mcc


def bootstrap_ci(y, p, n_boot=1000, seed=0, alpha=0.05):
    """95% percentile bootstrap CIs for accuracy, macro-F1 and MCC (resample test reviews with replacement)."""
    rng = np.random.default_rng(seed)
    pred = (p >= 0.5).astype(np.int8)
    y = y.astype(np.int8)
    stats = {"accuracy": [], "f1_macro": [], "mcc": []}
    for start in range(0, n_boot, 100):  # chunks keep memory small
        idx = rng.integers(0, len(y), size=(min(100, n_boot - start), len(y)))
        yy, pp = y[idx], pred[idx]
        tp = ((yy == 1) & (pp == 1)).sum(1).astype(float)
        tn = ((yy == 0) & (pp == 0)).sum(1).astype(float)
        fp = ((yy == 0) & (pp == 1)).sum(1).astype(float)
        fn = ((yy == 1) & (pp == 0)).sum(1).astype(float)
        for k, v in zip(stats, _counts_metrics(tp, fp, fn, tn)):
            stats[k].append(v)
    out = {}
    for k, v in stats.items():
        v = np.concatenate(v)
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return out


def mcnemar(y, p_base, p_exp):
    """Paired McNemar test on the same test reviews.
    b = baseline right & experimental wrong, c = baseline wrong & experimental right."""
    base_ok = (p_base >= 0.5).astype(int) == y
    exp_ok = (p_exp >= 0.5).astype(int) == y
    b, c = int((base_ok & ~exp_ok).sum()), int((~base_ok & exp_ok).sum())
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0  # with continuity correction
    return {"b_base_only_correct": b, "c_exp_only_correct": c, "chi2": stat,
            "p_value": float(chi2.sf(stat, 1)), "p_value_exact": binomtest(b, b + c).pvalue if b + c else 1.0}


def slice_metrics(y, p, slice_labels):
    """Macro-F1 and error rate for each value of a slice."""
    pred = (p >= 0.5).astype(int)
    rows = []
    for name, labels in slice_labels.items():
        for value in sorted(set(labels)):
            m = labels == value
            rows.append({"slice": name, "value": value, "n": int(m.sum()),
                         "f1_macro": f1_score(y[m], pred[m], average="macro", zero_division=0),
                         "error_rate": float((pred[m] != y[m]).mean())})
    return rows


# ---------------- plots ----------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion(y, p, title, path):
    plt = _plt()
    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["negative", "positive"], yticklabels=["negative", "positive"],
           xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(y, probs, path):
    """ROC, precision-recall and reliability (calibration) curves for all models."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.4f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.4f})")
        ece, bins = expected_calibration_error(y, p)
        axes[2].plot([b[4] for b in bins], [b[3] for b in bins], "o-", label=f"{name} (ECE {ece:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[2].plot([0.5, 1], [0.5, 1], "k--", lw=0.8, label="perfect calibration")
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall")
    axes[2].set(xlabel="confidence (prob. of predicted class)", ylabel="accuracy", title="Reliability diagram")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(histories, path):
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        a1.plot(ep, [r["train_loss"] for r in h], "o--", label=f"{name} train")
        a1.plot(ep, [r["val_loss"] for r in h], "o-", label=f"{name} val")
        a2.plot(ep, [r["val_acc"] for r in h], "o-", label=name)
    a1.set(xlabel="epoch", ylabel="BCE loss", title="Training / validation loss")
    a2.set(xlabel="epoch", ylabel="accuracy", title="Validation accuracy")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
