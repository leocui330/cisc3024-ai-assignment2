"""Hard-label probability metrics, independent of the training objective."""
import numpy as np


def metrics(probabilities, labels, bins=15):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2 or len(p) != len(y) or len(y) == 0:
        raise ValueError("Invalid prediction shape")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=2e-6):
        raise ValueError("Predictions must be finite normalized probabilities")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("Label outside class range")
    pred, confidence = p.argmax(1), p.max(1)
    correct = pred == y
    true_probability = p[np.arange(len(y)), y]
    clipped_count = int((true_probability < 1e-15).sum())
    nll = -np.log(np.clip(true_probability, 1e-15, 1)).mean()
    onehot = np.eye(p.shape[1])[y]
    brier = np.square(p - onehot).sum(1).mean()
    bin_id = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability, ece = [], 0.0
    for k in range(bins):
        mask = bin_id == k
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else None
        conf = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(acc - conf)
        reliability.append(dict(bin=k, count=count, accuracy=acc, confidence=conf))
    return dict(accuracy=float(correct.mean()*100), nll=float(nll), brier=float(brier),
                ece=float(ece*100), n=len(y), reliability=reliability,
                clipped_true_probability_count=clipped_count,
                minimum_true_probability=float(true_probability.min()))
