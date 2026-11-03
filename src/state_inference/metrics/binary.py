"""Binary metrics implemented without adding heavyweight dependencies."""
import math


def binary_metrics(labels, scores):
    if not labels or len(labels) != len(scores):
        raise ValueError("Nonempty matching labels/scores required")
    if any(y not in (0, 1) for y in labels) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("Invalid labels/probabilities")
    n = len(labels)
    positives = sum(labels)
    negatives = n - positives
    tp = sum(y == 1 and p >= 0.5 for y, p in zip(labels, scores))
    fp = sum(y == 0 and p >= 0.5 for y, p in zip(labels, scores))
    fn = positives - tp
    # Pairwise AUROC, ties count 1/2. Small pilot subsets only.
    roc = (sum((p > q) + 0.5 * (p == q) for y, p in zip(labels, scores) if y == 1
               for z, q in zip(labels, scores) if z == 0) / (positives * negatives)
           if positives and negatives else None)
    # Threshold-grouped average precision; robust to score ties.
    found, seen, ap = 0, 0, 0.0
    for value in sorted(set(scores), reverse=True):
        block = [y for y, p in zip(labels, scores) if p == value]
        added = sum(block)
        found += added
        seen += len(block)
        ap += added * found / seen
    eps = 1e-7
    return {"n": n, "positives": positives, "negatives": negatives,
            "auroc": roc, "average_precision": ap / positives if positives else None,
            "f1_at_0_5": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy_at_0_5": sum((p >= 0.5) == bool(y) for y, p in zip(labels, scores)) / n,
            "brier": sum((p-y)**2 for y, p in zip(labels, scores)) / n,
            "bce": -sum(y*math.log(max(eps, p)) + (1-y)*math.log(max(eps, 1-p)) for y, p in zip(labels, scores)) / n}
