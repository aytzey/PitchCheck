"""NumPy-only WS08 inference checks; CI does not need scipy, an atlas or GPU."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("ws08_stats", Path(__file__).resolve().parents[2] / "scripts/research/stats.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def test_planted_and_null_signflip():
    rng = np.random.default_rng(4)
    assert stats.signflip_p(rng.normal(.8, 1, 20)) < .05
    false_positive = np.mean([stats.signflip_p(rng.normal(size=20), B=999, seed=i) < .05 for i in range(500)])
    assert .02 <= false_positive <= .09


def test_exact_and_monte_carlo_and_smallest_p():
    d = np.random.default_rng(7).normal(.2, 1, 16)
    assert abs(stats.signflip_p(d) - stats.signflip_p(np.r_[d, 0], B=30_000)) < .015
    assert stats.signflip_p([1, 1, 1, 1]) == 1 / 16
    assert stats.signflip_p([-1] * 4, "less") == 1 / 16
    assert stats.signflip_p([1] * 4, "two-sided") == 2 / 16
    assert stats.signflip_p([0] * 20) == 1


def test_parcel_rank_smooth_synthetic_block_null():
    # Each column is a synthetic parcel block; nearby parcels have smooth effects.
    maps = np.random.default_rng(3).normal(size=(500, 400))
    smooth = sum(np.roll(maps, shift, axis=1) for shift in range(-8, 9))
    rate = np.mean([stats.parcel_rank(row, row[0]) >= .90 for row in smooth])
    assert .06 <= rate <= .14
    assert stats.parcel_rank([1, 2, 3, 4], 2, "less") == .5


def test_icc_variance_components_and_degenerate_case():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2, (2000, 1)) + rng.normal(size=(2000, 5))
    icc, sigma = stats.icc1(x)
    assert icc == pytest.approx(.8, abs=.05)
    assert sigma == pytest.approx(1, abs=.05)
    assert stats.icc1(np.tile(np.arange(10)[:, None], (1, 5))) == (1.0, 0.0)
    assert stats.icc1(np.ones((3, 5))) == (0.0, 0.0)


def test_holm_dz_residuals_bootstrap():
    assert stats.holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])
    assert stats.dz([1, 2, 3]) == 2
    assert stats.dz([0, 0]) == 0
    x = np.arange(5.)
    np.testing.assert_allclose(stats.residualize(3 + 2 * x, x), 0, atol=1e-12)
    assert stats.boot_ci([2, 2, 2]) == [2, 2]
    assert stats.boot_ci(x) == stats.boot_ci(x)


@pytest.mark.parametrize("call", [lambda: stats.signflip_p([]), lambda: stats.signflip_p([np.nan]),
    lambda: stats.signflip_p([1], "invalid"), lambda: stats.holm([1.1]), lambda: stats.icc1([[1]]),
    lambda: stats.residualize([1, 2], [1]), lambda: stats.parcel_rank([], 1)])
def test_invalid_samples_fail_explicitly(call):
    with pytest.raises(ValueError):
        call()
