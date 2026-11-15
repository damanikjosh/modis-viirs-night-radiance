import pytest
import torch

from nightrad.raw.hide import HideConfig, hide_regions
from nightrad.raw.mask import Footprints, common_mask, dense_pid, roi_mask, valid_fraction_ok
from nightrad.raw.views import ModisView, ViewConfig


def _grid_pid(b=2, h=12, w=12, fh=3, fw=4):
    r = torch.arange(h)[:, None] // fh
    c = torch.arange(w)[None, :] // fw
    pid = (r * (w // fw) + c).expand(b, h, w).clone()
    return pid.long()


def test_roi_and_common_mask_and_widening():
    cm = torch.zeros(1, 9, 9, dtype=torch.uint8); cv = cm.clone()
    cm[0, 4, 4] = 1  # cloud
    cv[0, 0, :] = 4  # not covered
    assert not roi_mask(cm)[0, 4, 4] and roi_mask(cm)[0, 4, 5]
    w1 = roi_mask(cm, widen=1)
    assert not w1[0, 3:6, 3:6].any() and w1[0, 4, 6]  # d=1 removes the 3x3 neighbourhood only
    com = common_mask(cm, cv)
    assert not com[0, 0].any() and not com[0, 4, 4] and com[0, 8, 8]
    assert valid_fraction_ok(com, 0.5)[0]


def test_footprint_ok_requires_every_cell_and_complete():
    pid = _grid_pid(b=1)
    fp = Footprints(pid)
    common = torch.ones(1, 12, 12, dtype=torch.bool)
    complete = torch.ones(1, 12, 12, dtype=torch.uint8)
    common[0, 0, 0] = False  # one cell out -> its whole footprint is out
    complete[0, 11, 11] = 0
    ok = fp.ok(common, complete)
    assert not ok[0, :3, :4].any() and not ok[0, 9:, 8:].any()
    assert ok[0, 3:6, 4:8].all()


def test_modis_view_of_constant_scene_returns_the_constant():
    pid = _grid_pid()
    fp = Footprints(pid)
    B = torch.full((2, 2, 12, 12), 0.5)
    B[:, 1] = 9.0
    mv = ModisView()
    out = mv(B, torch.zeros_like(B), fp)
    # background only: constant in, planck-converted constant out, identical in every cell
    assert torch.allclose(out, out[:, :, :1, :1].expand_as(out), atol=1e-7)
    from nightrad.raw.planck import convert_background
    assert float(out[0, 0, 0, 0]) == pytest.approx(float(convert_background(torch.tensor(0.5), 3.74, 3.96)), rel=1e-6)
    # excess-only constant scene: view = factor x constant
    E = torch.full_like(B, 0.2)
    ex = mv(torch.full_like(B, 1e-9), E, fp) - mv(torch.full_like(B, 1e-9), torch.zeros_like(E), fp)
    assert float(ex[0, 0, 0, 0]) == pytest.approx(mv.f4 * 0.2, rel=1e-5)


def test_energy_conservation_residual_zero_on_random_scenes():
    torch.manual_seed(0)
    # irregular groups: random pids
    pid = _grid_pid(b=3)
    pid = (pid + (torch.rand(pid.shape) < 0.1).long() * 7) % 144  # irregularity
    fp = Footprints(pid)
    mv = ModisView()
    E = torch.rand(3, 2, 12, 12) ** 3
    view_excess = mv.excess(E, fp)  # per-footprint excess radiance (per cell broadcast)
    f = torch.tensor([mv.f4, mv.f11])[None, :, None, None]
    # sum over each footprint's cells of E_c (equal cell area) x factor  ==  footprint view excess x n cells
    tot = fp.total(E) * f
    n = fp.total(torch.ones_like(E))
    resid = (tot - view_excess * n).abs().max()
    assert float(resid) < 1e-5


def test_logsumexp_and_allocation_softmax_sums_to_one():
    pid = _grid_pid(b=2)
    fp = Footprints(pid)
    x = torch.randn(2, 12, 12) * 3
    p = torch.exp(x - fp.logsumexp(x))
    assert torch.allclose(fp.total(p[:, None])[:, 0], torch.ones(2, 12, 12), atol=1e-5)


def test_hiding_selects_fire_windows_hides_whole_pixels_and_stays_in_mask():
    torch.manual_seed(0)
    rho = torch.zeros(4, 24, 24)
    rho[:, 12, 12] = 2.0
    valid = torch.ones(4, 24, 24, dtype=torch.bool)
    valid[:, :, :3] = False
    cfg = HideConfig(p_tile=1.0, n_random=0, n_top=1)
    h = hide_regions(rho, valid, cfg, torch.Generator().manual_seed(0))
    assert h[:, 8:17, 8:17].all() and h.sum() == 4 * 81  # 9x9 window on the fire cell
    assert not (h & ~valid).any()
    # tiles gated by p_tile
    assert hide_regions(rho, valid, HideConfig(p_tile=0.0), torch.Generator().manual_seed(0)).sum() == 0
    # MODIS: hiding any cell hides the whole native pixel
    pid = _grid_pid(b=4, h=24, w=24, fh=5, fw=7)
    hm = hide_regions(rho, valid, HideConfig(p_tile=1.0, n_random=0, n_top=1, window=3), torch.Generator().manual_seed(0), Footprints(pid))
    for b in range(4):
        for g in pid[b][hm[b]].unique():
            assert hm[b][(pid[b] == g) & valid[b]].all()


def test_partial_hiding_covers_part_only():
    rho = torch.zeros(64, 24, 24); rho[:, 12, 12] = 2.0
    valid = torch.ones(64, 24, 24, dtype=torch.bool)
    full = hide_regions(rho, valid, HideConfig(p_tile=1.0, n_random=0, n_top=1, partial_p=0.0), torch.Generator().manual_seed(1))
    part = hide_regions(rho, valid, HideConfig(p_tile=1.0, n_random=0, n_top=1, partial_p=1.0, jitter=3), torch.Generator().manual_seed(1))
    assert part.sum() < full.sum() and part.sum() > 0
    assert (part[:, 12, 12].float().mean() < 1.0)  # sometimes the fire cell itself is left visible
