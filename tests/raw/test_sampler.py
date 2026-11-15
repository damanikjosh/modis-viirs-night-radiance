"""NIGHT: region-balanced sampling and the fire-tiles-in-every-batch sampler (flags from the MODIS input only)."""
import numpy as np
import torch

from nightrad.raw.data import FireBatchSampler, region_balanced_weights


def test_region_weights_equalise_regions():
    reg = np.array([0] * 900 + [1] * 90 + [2] * 10)
    w = region_balanced_weights(reg)
    assert abs(w.sum() - 1.0) < 1e-9
    for r in range(3):
        assert abs(w[reg == r].sum() - 1 / 3) < 1e-9
    draws = torch.multinomial(torch.as_tensor(w), 30000, replacement=True, generator=torch.Generator().manual_seed(0)).numpy()
    share = np.bincount(reg[draws], minlength=3) / len(draws)
    assert np.allclose(share, 1 / 3, atol=0.02)


def test_fire_batches_hold_fire_tiles_and_keep_regions():
    rng = np.random.default_rng(0)
    reg = rng.integers(0, 4, 2000); fire = rng.random(2000) < 0.1
    w = region_balanced_weights(reg)
    s = FireBatchSampler(w, fire, 32, 0.25, seed=3)
    assert len(s) == 2000 // 32
    batches = list(s)
    assert all(len(b) == 32 for b in batches)
    nf = np.array([fire[b].sum() for b in batches])
    assert (nf >= 8).all()                     # 8 forced fire tiles; the other 24 add on average 10 % more
    assert nf.mean() < 8 + 24 * 0.1 + 1.0
    allidx = np.concatenate(batches); share = np.bincount(reg[allidx], minlength=4) / len(allidx)
    assert np.allclose(share, 0.25, atol=0.06)  # regions stay roughly balanced (fire pool is region-balanced too)
    assert list(FireBatchSampler(w, fire, 32, 0.25, seed=3)) == batches          # deterministic
    s.set_epoch(1); assert list(s) != batches                                   # reseeded per epoch


def test_no_fire_tiles_falls_back():
    s = FireBatchSampler(np.ones(100), np.zeros(100, bool), 10, 0.5)
    assert s.nf == 0 and all(len(b) == 10 for b in s)
