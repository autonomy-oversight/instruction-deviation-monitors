"""Detector metrics shared by both pilot parts."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(labels, scores):
    """AUROC, or None when only one class is present."""
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(labels, scores, max_fpr=0.10):
    """Detection rate at a fixed false-alarm rate (audit budget)."""
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def bootstrap_ci(labels, scores, fn=auroc, n=1000, seed=0):
    """95% bootstrap interval for a metric; small pilot samples need it."""
    rng = np.random.default_rng(seed)
    labels, scores = np.asarray(labels), np.asarray(scores)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        v = fn(labels[idx], scores[idx])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def report(name, labels, scores, max_fpr=0.10, show_tpr=True):
    """AUROC with bootstrap CI; optionally the in-sample best detection rate at the given false-alarm rate."""
    a = auroc(labels, scores)
    if a is None:
        return f"{name}: only one class present (n={len(labels)}), metric undefined"
    ci = bootstrap_ci(labels, scores)
    pos = int(np.sum(labels))
    out = f"{name}: n={len(labels)} (deviations={pos}) | AUROC={a:.2f} [95% CI {ci[0]:.2f}-{ci[1]:.2f}]"
    if show_tpr:
        out += f" | in-sample detection at FPR<={max_fpr:.0%}: {tpr_at_fpr(labels, scores, max_fpr):.2f}"
    return out
