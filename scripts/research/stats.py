"""WS08 paired statistics. NumPy only; no inference over human subjects."""
from __future__ import annotations

import numpy as np


def _sample(values, minimum=1):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < minimum or not np.isfinite(values).all():
        raise ValueError("Expected a finite, nonempty one-dimensional sample")
    return values


def signflip_p(d, alternative="greater", B=10_000, seed=0):
    d = _sample(d)
    if alternative not in {"greater", "less", "two-sided"} or B < 1:
        raise ValueError("Invalid alternative or permutation count")
    statistic = np.abs if alternative == "two-sided" else (lambda x: -x if alternative == "less" else x)
    if len(d) <= 16:
        signs = 1 - 2 * ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1)
    else:
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(B, len(d)))
    hits = np.count_nonzero(statistic((signs * d).mean(1)) >= statistic(d.mean()) - 1e-12)
    return float(hits / len(signs) if len(d) <= 16 else (1 + hits) / (B + 1))


def dz(d):
    d = _sample(d, 2)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else (float(np.copysign(np.inf, d.mean())) if d.mean() else 0.0)


def boot_ci(d, B=2000, seed=0):
    d = _sample(d)
    if B < 1:
        raise ValueError("Bootstrap count must be positive")
    means = np.random.default_rng(seed).choice(d, size=(B, len(d))).mean(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def holm(p):
    p = _sample(p)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p values must be in [0, 1]")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def icc1(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("ICC needs at least two complete families and two paraphrases")
    k = x.shape[1]
    msb, msw = k * x.mean(1).var(ddof=1), x.var(1, ddof=1).mean()
    denominator = msb + (k - 1) * msw
    # A constant quantity has no measurable reliability; do not call it stable.
    return (float((msb - msw) / denominator) if denominator else 0.0), float(np.sqrt(msw))


def parcel_rank(dz_parcels, dz_target, alternative="greater"):
    values = np.asarray(dz_parcels, dtype=float)
    if values.ndim != 1 or not len(values) or np.isnan(values).any() or np.isnan(dz_target):
        raise ValueError("Expected parcel effects and a target effect")
    if alternative not in {"greater", "less"}:
        raise ValueError("Invalid rank direction")
    return float(np.mean(values < dz_target) if alternative == "greater" else np.mean(values > dz_target))


def residualize(y, X):
    y, X = np.asarray(y, float), np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if y.ndim != 1 or X.ndim != 2 or len(y) != len(X) or not np.isfinite(y).all() or not np.isfinite(X).all():
        raise ValueError("Invalid regression data")
    design = np.column_stack([np.ones(len(y)), X])
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
