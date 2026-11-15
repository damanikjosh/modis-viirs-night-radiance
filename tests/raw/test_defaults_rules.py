"""Defaults of the run configuration (weighted view, night only, group-share loss, purpose-weighted terms), the detection rule, the test-split guard."""
import json
import subprocess
import sys

import torch

from nightrad.raw import planck
from nightrad.raw.core import Prep, RunConfig, forward_loss
from nightrad.raw.data import make_loader
from nightrad.raw.losses import LossConfig, shares_for_fire
from nightrad.raw.mask import Footprints
from nightrad.raw.net import NetConfig, RawHarmonizer
from nightrad.raw.rules import detect
from nightrad.raw.synth import SyntheticTiles
from nightrad.raw.views import ModisView, ViewConfig
from raw_helpers import tiny_cfg


def test_defaults_are_the_approved_ones():
    c = RunConfig()
    assert c.view.kernel == "tri" and c.daynight == "night" and c.loss.class_balance
    assert c.loss.class_shares == (0.4, 0.2, 0.4)
    assert c.loss.term_weights == {"mv": 1.0, "vm": 0.5, "mm": 0.1, "vv": 0.1}
    assert c.net.arch == "unet" and not c.net.hurdle_head and c.mv_mode == "free"  # deterministic, no hurdle/allocation
    for f in (0.2, 0.4, 0.6):
        s = shares_for_fire(f)
        assert abs(sum(s) - 1) < 1e-9 and s[2] == f and abs(s[0] - 2 * s[1]) < 1e-9


def test_night_default_drops_day_tiles_and_group_shares_are_logged(small_batch):
    inp, _ = small_batch
    P = Prep(inp, RunConfig())
    day = P.day
    assert day.any() and (~day).any()
    assert not (P.common & day[:, None, None]).any() and (P.common & ~day[:, None, None]).any()
    m = RawHarmonizer(NetConfig(width=8, latent=4))
    _, logs = forward_loss(m, inp, RunConfig(net=NetConfig(width=8, latent=4)), torch.Generator().manual_seed(0))
    for t in ("mv", "vm", "mm", "vv"):
        assert all(f"{t}/share_{g}" in logs for g in ("bg", "near", "fire"))


def test_weighted_view_spreads_into_neighbours_and_conserves_unit_response(small_batch):
    inp, _ = small_batch
    cfg = tiny_cfg(view=ViewConfig(kernel="tri", w_scan=0.125, w_track=0.05))
    P = Prep(inp, cfg)
    B = torch.ones(8, 2, 48, 48); E = torch.zeros_like(B)
    mv_t, mv_b = ModisView(cfg.view), ModisView(ViewConfig(kernel="box"))
    assert torch.allclose(mv_t(B, E, P.fp), mv_b(B, E, P.fp), atol=1e-6)  # constant scene: weights sum to 1
    E[:, 0, 10:13, 10:13] = 1.0  # a hot spot spreads into neighbouring pixels in the weighted view only
    t, b = mv_t(B, E, P.fp)[:, 0], mv_b(B, E, P.fp)[:, 0]
    assert (t - b).abs().max() > 1e-4 and abs(float(t.sum() - b.sum())) / float(b.sum()) < 0.2


def test_rule_applies_same_code_and_finds_a_fire_not_a_flat_scene():
    L4 = torch.full((1, 24, 24), float(planck.planck_radiance(290.0, 3.74))); L11 = torch.full((1, 24, 24), float(planck.planck_radiance(285.0, 11.45)))
    valid = torch.ones(1, 24, 24, dtype=torch.bool)
    night = torch.zeros(1, dtype=torch.bool)
    assert not detect(L4, L11, valid, night, 3.74, 11.45).any()
    L4h = L4.clone(); L4h[0, 12, 12] = float(planck.planck_radiance(340.0, 3.74))
    d = detect(L4h, L11, valid, night, 3.74, 11.45)
    assert d[0, 12, 12] and d.sum() == 1


def test_block_loss_is_off_by_default_and_trains_when_on(small_batch):
    from nightrad.raw.losses import LossConfig
    inp, _ = small_batch
    torch.manual_seed(0)
    m = RawHarmonizer(NetConfig(width=8, latent=4))
    c0 = tiny_cfg(); c1 = tiny_cfg(loss=LossConfig(block_w_mv=1.0, block_w_vm=1.0))
    l0, g0 = forward_loss(m, inp, c0, torch.Generator().manual_seed(0))
    l1, g1 = forward_loss(m, inp, c1, torch.Generator().manual_seed(0))
    assert "block_mv/loss" not in g0 and "block_mv/loss" in g1 and "block_vm/loss" in g1
    l1.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert float(l1) > float(l0) - 1e-9


def test_test_split_guard_checks_the_data_not_the_label(tmp_path):
    import json, os
    from nightrad.raw import prereg
    real = tmp_path / "test_region"; real.mkdir()
    link = tmp_path / "val"; os.symlink(real, link)               # innocent name, symlink to a test folder
    assert prereg.is_test_split(str(link))
    lab = tmp_path / "val_night"; lab.mkdir(); json.dump({"split": "test"}, open(lab / "pack_report.json", "w"))  # innocent name, test data inside
    assert prereg.is_test_split(str(lab))
    ok = tmp_path / "val_ok"; ok.mkdir(); json.dump({"split": "val"}, open(ok / "pack_report.json", "w"))
    assert not prereg.is_test_split(str(ok))
    import pytest
    with pytest.raises(PermissionError):
        prereg.open_split(str(lab))
