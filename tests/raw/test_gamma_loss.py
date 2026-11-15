"""toy with the REAL loss code: one MODIS pixel = 9 fine cells, background 1, one cell burns at 21x, position unknown, truth total 29.
Energy-conserving value = 29/9 = 3.222 per cell. Reference numbers of the toy: current (group shares + Huber on log) 20.65, recommended Gamma deviance 3.22."""
import math
import torch
from nightrad.raw.losses import LossConfig, gamma_dev, gamma_term, term_loss
from nightrad.raw.net import NetConfig, RawHarmonizer
from nightrad.raw.core import forward_loss
from raw_helpers import tiny_cfg

H, K, FIRE, BG = 16, 3, 21.0, 1.0


def _target(pos):
    y = torch.full((1, 2, H, H), BG); r0 = H // 2 - 1; py, px = r0 + pos // K, r0 + pos % K
    y[0, :, py, px] = FIRE
    return y, (py, px)


def _legacy_loss(c, cfg):
    tot = 0.0
    for pos in range(K * K):
        y, (py, px) = _target(pos); tgt = torch.log(y); rho = torch.zeros(1, H, H); rho[0, py, px] = math.log(FIRE)
        pred = tgt.clone(); r0 = H // 2 - 1; pred[0, :, r0:r0 + K, r0:r0 + K] = math.log(c)
        l, _ = term_loss(pred, tgt, torch.ones(1, H, H, dtype=torch.bool), torch.ones(1, H, H), torch.zeros(1, H, H, dtype=torch.bool), cfg, rho)
        tot += float(l)
    return tot / (K * K)


def _gamma_loss(c, cfg):
    tot = 0.0
    for pos in range(K * K):
        y, _ = _target(pos); mu = y.clone(); r0 = H // 2 - 1; mu[:, :, r0:r0 + K, r0:r0 + K] = c
        blk = torch.zeros(1, H, H, dtype=torch.bool); blk[0, r0:r0 + K, r0:r0 + K] = True  # score the pixel's own cells (the rest is exact)
        tot += float(gamma_term(mu, y, blk, torch.ones(1, H, H), torch.zeros(1, H, H, dtype=torch.bool), cfg))
    return tot / (K * K)


def _argmin(f, lo=0.5, hi=40.0, n=4000):
    g = torch.exp(torch.linspace(math.log(lo), math.log(hi), n)); vals = torch.tensor([f(float(c)) for c in g[::8]])
    c0 = float(g[::8][vals.argmin()]); g2 = torch.exp(torch.linspace(math.log(c0 / 1.05), math.log(c0 * 1.05), 400)); v2 = torch.tensor([f(float(c)) for c in g2])
    return float(g2[v2.argmin()])


def test_gamma_deviance_basic():
    y = torch.tensor([1.0, 2.0, 5.0]); assert float(gamma_dev(y, y).abs().max()) < 1e-6
    assert float(gamma_dev(y * 2, y).min()) > 0 and torch.isfinite(gamma_dev(torch.zeros(3), y)).all()


def test_toy_current_loss_over_paints_recommended_loss_conserves_energy():
    true_total = FIRE + (K * K - 1) * BG
    cur = _argmin(lambda c: _legacy_loss(c, LossConfig()))
    gam = _argmin(lambda c: _gamma_loss(c, LossConfig(mode="gamma")))
    assert 19.0 < cur < 22.0, cur                      # 20.65 per cell = 6.4x the energy
    assert abs(cur * K * K / true_total - 6.4) < 0.7
    assert abs(gam - true_total / (K * K)) < 0.05, gam   # 3.222 per cell = 1.00x the energy
    assert abs(gam * K * K / true_total - 1.0) < 0.02


def test_gamma_mode_runs_trains_and_has_no_target_weights(small_batch):
    inp, _ = small_batch
    torch.manual_seed(0); m = RawHarmonizer(NetConfig(width=8, latent=4))
    cfg = tiny_cfg(loss=LossConfig(mode="gamma", w_energy=0.1))
    l, logs = forward_loss(m, inp, cfg, torch.Generator().manual_seed(0))
    assert torch.isfinite(l) and "mv/cell" in logs and "mv/pix" in logs and "energy/loss" in logs
    l.backward(); assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


# ---- three loss families gamma / poisson / nmse must all be mean-consistent in the toy; L1 and log-L2 stay as failing references
import pytest


def _family_loss(c, family):
    cfg = LossConfig(mode="gamma", loss_family=family); tot = 0.0
    r0 = H // 2 - 1; ref = torch.tensor([1.0, 1.0]); bg = torch.ones(1, 2, H, H)
    for pos in range(K * K):
        y, _ = _target(pos); mu = y.clone(); mu[:, :, r0:r0 + K, r0:r0 + K] = c
        blk = torch.zeros(1, H, H, dtype=torch.bool); blk[0, r0:r0 + K, r0:r0 + K] = True
        tot += float(gamma_term(mu, y, blk, torch.ones(1, H, H), torch.zeros(1, H, H, dtype=torch.bool), cfg, ref=ref, bg=bg))
    return tot / (K * K)


@pytest.mark.parametrize("family", ["gamma", "poisson", "nmse"])
def test_each_family_minimiser_is_the_mean_energy_ratio_one(family):
    true_total = FIRE + (K * K - 1) * BG
    c = _argmin(lambda v: _family_loss(v, family))
    assert abs(c * K * K / true_total - 1.0) <= 0.02, (family, c)


def test_l1_and_log_l2_are_not_mean_consistent_reference():
    true_total = FIRE + (K * K - 1) * BG
    l1 = _argmin(lambda c: sum(abs(torch.tensor(c) - _target(p)[0]).mean().item() for p in range(K * K)))  # median -> deletes the fire
    ll2 = _argmin(lambda c: sum(((math.log(c) - torch.log(_target(p)[0])) ** 2).mean().item() for p in range(K * K)))  # geometric mean
    assert l1 * K * K / true_total < 0.5 and ll2 * K * K / true_total < 0.6


def test_families_have_no_target_dependent_weights_and_floor(small_batch):
    inp, _ = small_batch
    for fam in ("gamma", "poisson", "nmse"):
        torch.manual_seed(0); m = RawHarmonizer(NetConfig(width=8, latent=4))
        l, logs = forward_loss(m, inp, tiny_cfg(loss=LossConfig(mode="gamma", loss_family=fam, w_energy=0.1)), torch.Generator().manual_seed(0))
        assert torch.isfinite(l), fam
        l.backward(); assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None), fam
