"""Sweep-1 additions (review C1/H1): energy-consistency term, false-excess loss group, hiding-window centre jitter, resume key."""
import torch
from nightrad.raw.core import Prep, forward_loss
from nightrad.raw.hide import HideConfig, hide_regions
from nightrad.raw.losses import LossConfig, term_loss
from nightrad.raw.net import NetConfig, RawHarmonizer
from raw_helpers import tiny_cfg


def test_energy_term_off_by_default_adds_loss_and_trains(small_batch):
    inp, _ = small_batch
    torch.manual_seed(0); m = RawHarmonizer(NetConfig(width=8, latent=4))
    l0, g0 = forward_loss(m, inp, tiny_cfg(), torch.Generator().manual_seed(0))
    l1, g1 = forward_loss(m, inp, tiny_cfg(loss=LossConfig(w_energy=1.0)), torch.Generator().manual_seed(0))
    assert "energy/loss" not in g0 and "energy/loss" in g1 and float(g1["energy/loss"]) >= 0
    assert float(l1) >= float(l0) - 1e-9
    l1.backward(); assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_false_excess_group_is_a_fourth_share():
    cfg = LossConfig(false_share=0.2)
    mask = torch.ones(1, 12, 12, dtype=torch.bool); rho = torch.zeros(1, 12, 12); rho[0, 2, 2] = 1.0
    fm = torch.zeros(1, 12, 12, dtype=torch.bool); fm[0, 6:9, 6:9] = True  # a bright MODIS pixel without VIIRS fire
    pred = torch.zeros(1, 2, 12, 12); pred[:, :, 6:9, 6:9] = 0.5  # smeared
    tgt = torch.zeros(1, 2, 12, 12)
    l, st = term_loss(pred, tgt, mask, torch.ones(1, 12, 12), torch.zeros_like(mask), cfg, rho, fm)
    tot = sum(float(st[f"share_{n}"]) for n in ("bg", "near", "fire", "false"))
    assert abs(tot - 1) < 1e-5 and float(st["share_false"]) > 0.9  # only the smeared cells have error, all of it in the false group
    l0, st0 = term_loss(pred, tgt, mask, torch.ones(1, 12, 12), torch.zeros_like(mask), LossConfig(), rho, fm)
    assert "share_false" not in st0 and float(l) > float(l0)  # penalised explicitly


def test_hiding_centre_jitter_moves_the_hole_but_keeps_the_fire_inside():
    rho = torch.zeros(1, 40, 40); rho[0, 20, 20] = 2.0; valid = torch.ones(1, 40, 40, dtype=torch.bool)
    cents = set()
    for s in range(30):
        h = hide_regions(rho, valid, HideConfig(p_tile=1.0, n_top=1, n_random=0, center_jitter=3), torch.Generator().manual_seed(s))
        assert h[0, 20, 20]  # window 9 (half 4) always still covers the fire when the shift is <= 3
        ys, xs = h[0].nonzero().T; cents.add((int(ys.float().mean().round()), int(xs.float().mean().round())))
    assert len(cents) > 5
    h0 = hide_regions(rho, valid, HideConfig(p_tile=1.0, n_top=1, n_random=0), torch.Generator().manual_seed(0))
    ys, xs = h0[0].nonzero().T; assert (int(ys.float().mean().round()), int(xs.float().mean().round())) == (20, 20)


def test_region_balanced_weights_equalise_regions_and_loader_samples_them():
    import numpy as np
    from nightrad.raw.data import region_balanced_weights
    ids = np.array([0] * 90 + [1] * 5 + [2] * 5)                 # one dominant region (e.g. Australia)
    w = region_balanced_weights(ids)
    share = [w[ids == r].sum() / w.sum() for r in (0, 1, 2)]
    assert all(abs(s - 1 / 3) < 1e-9 for s in share)
    rng = np.random.default_rng(0); draw = rng.choice(len(ids), 30000, p=w / w.sum())
    assert abs((ids[draw] == 0).mean() - 1 / 3) < 0.02
