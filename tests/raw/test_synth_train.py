"""Synthetic scenes with known truth: the physics baseline, a short training run that recovers the excess, and degenerate-solution checks."""
import copy
import json
import os
from argparse import Namespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from nightrad.raw.core import Prep, RunConfig, forward_loss
from nightrad.raw.data import make_loader
from nightrad.raw.metrics import Acc, check_degenerate, evaluate_batch, excess_collapsed, physical_pred_modis, summarize
from nightrad.raw.net import NetConfig, RawHarmonizer
from nightrad.raw.schema import split_batch
from nightrad.raw.synth import SyntheticTiles, make_tile
from nightrad.raw.train import train


def test_synthetic_tile_schema_and_truth():
    d = make_tile(3, 48, 48)
    for s in ("modis", "viirs"):
        assert d[f"{s}_L"].shape == (2, 48, 48) and d[f"{s}_code"].dtype == np.uint8
    assert d["modis_pid"].min() >= 0 and d["modis_nbr"].shape == (4, 48, 48) and d["modis_complete"].dtype == np.uint8
    pid, L = d["modis_pid"], d["modis_L"][0]
    for g in np.unique(pid)[:20]:  # constant within a footprint
        v = L[pid == g]
        v = v[np.isfinite(v)]
        assert v.size == 0 or np.ptp(v) < 1e-6 * max(abs(v.mean()), 1)
    ds = [make_tile(i, 48, 48) for i in range(12)]
    assert any(((t["viirs_code"] == 0) ^ (t["modis_code"] == 0)).any() for t in ds)  # partial overlap
    assert any((t["viirs_code"] == 1).any() for t in ds) and any(t["viirs_sat"].any() or t["modis_band21"].any() for t in ds)
    sizes = np.array([(float(t["modis_vza"].mean()), float(t["modis_psize"][0].mean())) for t in (make_tile(i, 48, 48) for i in range(30))])
    assert np.corrcoef(sizes[:, 0], sizes[:, 1])[0, 1] > 0.6  # footprints grow toward the swath edge


def test_physical_baseline_reproduces_modis_within_noise():
    """Average VIIRS over footprints (+ band conversion) vs the noisy MODIS measurement on background footprints, with known truth."""
    b = next(iter(DataLoader(SyntheticTiles(24, 0, 48, 48), batch_size=24)))
    inp, ev = split_batch(b)
    cfg = RunConfig(daynight="both")
    P = Prep(inp, cfg)
    pred = torch.log(physical_pred_modis(P, cfg))
    bg = P.ok_foot & (P.rho4["modis"] < 0.3)
    r4 = (pred[:, 0] - P.logL["modis"][:, 0])[bg]; r11 = (pred[:, 1] - P.logL["modis"][:, 1])[bg]
    assert bg.sum() > 2000
    assert abs(float(r4.median())) < 0.015 and float(r4.abs().median()) < 0.03
    assert abs(float(r11.median())) < 0.01 and float(r11.abs().median()) < 0.012
    fire = P.ok_foot & (P.rho4["modis"] >= 0.5)
    assert fire.sum() > 20 and float((pred[:, 0] - P.logL["modis"][:, 0]).abs()[fire].median()) < 0.2


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("synth_run"))
    a = Namespace(out=out, epochs=6, bs=16, lr=4e-3, wd=1e-4, warmup=10, amp="off", channels_last=False, compile=False, workers=0, seed=0,
                  resume=False, deterministic=False, clip=1.0)
    cfg = RunConfig(net=NetConfig(width=12, latent=6), daynight="both")
    tr, va = SyntheticTiles(160, 0, 48, 48, cache=True), SyntheticTiles(64, 10**6, 48, 48, cache=True)
    best = train(a, tr, va, cfg, log=lambda *_: None)
    ck = torch.load(os.path.join(out, "best.pt"), weights_only=False)
    model = RawHarmonizer(cfg.net); model.load_state_dict(ck["model"]); model.eval()
    return model, cfg, va, best, out


def _summary(model, cfg, ds):
    acc = Acc()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for b in make_loader(ds, 32, 0):
            inp, ev = split_batch(b)
            _, _, aux = forward_loss(model, inp, cfg, gen, return_aux=True)
            evaluate_batch(aux, inp, cfg, acc, ev)
    return summarize(acc)


def test_short_training_recovers_the_excess(trained):
    """Metric: median |d ln L4| at fire-level MODIS footprints, V->M (the model never sees MODIS), vs the background-only predictor.
    Thresholds: <= 0.30 x background-only; excess mass within [0.5, 2] of the TRUTH and of the measurement; rank corr >= 0.4."""
    model, cfg, va, best, out = trained
    s = _summary(model, cfg, va)
    assert s["n_vm_fire"] > 40
    assert s["vm_fire"] <= 0.30 * s["vm_fire_bgonly"], s
    assert 0.5 <= s["excess_mass_ratio"] <= 2.0 and s["rank_corr"] >= 0.4, s
    assert 0.5 <= s["excess_mass_ratio_meas"] <= 2.0
    assert check_degenerate(s) == []
    rows = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))]
    assert rows[-1]["mm/loss"] >= 0 and "vm/share_fire" in rows[-1] and "val/fire_err" in rows[-1]
    assert open(os.path.join(out, "status.txt")).read().count("\n") == 1
    assert best["fire_err"] == min(r["val/fire_err"] for r in rows)  # best by the fire-excess metric, not total loss


def test_degenerate_solutions_are_detected(trained):
    model, cfg, va, _, _ = trained
    good = _summary(model, cfg, va)
    assert check_degenerate(good) == []
    m0 = copy.deepcopy(model)  # (a) excess == 0 everywhere
    with torch.no_grad():
        m0.decoder.head.bias[2:4] = -30.0; m0.decoder.head.conv.weight[2:4] = 0; m0.rho_gain.zero_()
    s0 = _summary(m0, cfg, va)
    f0 = check_degenerate(s0)
    assert any("background-only" in x for x in f0) and any("mass" in x for x in f0), f0
    assert excess_collapsed(s0) and not excess_collapsed(good)  # the excess-mass stop rule flags it
    m1 = copy.deepcopy(model)  # (b) context-free constant output that ignores its input (cannot inpaint, cannot explain fires)
    with torch.no_grad():
        for p in list(m1.encoder.parameters()) + list(m1.decoder.parameters()):
            p.zero_()
        m1.decoder.head.bias[2:4] = -30.0; m1.rho_gain.zero_()
    assert check_degenerate(_summary(m1, cfg, va))
    assert good["vv_hidden_bg11"] <= 0.7 * good["vv_hidden_bg11_tilemedian"]  # (c) inpainting uses context


def test_training_is_resumable_and_deterministic(tmp_path):
    cfg = RunConfig(net=NetConfig(width=8, latent=4), daynight="both")
    mk = lambda out, ep, resume, stop=None: Namespace(out=str(out), epochs=ep, bs=8, lr=2e-3, wd=1e-4, warmup=2, amp="off", channels_last=False, compile=False,
                                           workers=0, seed=3, resume=resume, deterministic=True, clip=1.0, stop_after=stop)
    tr, va = SyntheticTiles(16, 0, 48, 48, cache=True), SyntheticTiles(8, 10**6, 48, 48, cache=True)
    train(mk(tmp_path / "a", 4, False), tr, va, cfg, log=lambda *_: None)
    train(mk(tmp_path / "b", 4, False, stop=2), tr, va, cfg, log=lambda *_: None)
    train(mk(tmp_path / "b", 4, True), tr, va, cfg, log=lambda *_: None)  # resumed from epoch 2
    la = [json.loads(l) for l in open(tmp_path / "a" / "log.jsonl")]
    lb = [json.loads(l) for l in open(tmp_path / "b" / "log.jsonl")]
    assert [r["epoch"] for r in lb] == [0, 1, 2, 3]
    assert la[-1]["total"] == pytest.approx(lb[-1]["total"], rel=1e-5)
