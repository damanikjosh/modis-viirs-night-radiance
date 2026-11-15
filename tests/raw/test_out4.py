"""the 4 um-only output option (NetConfig.out4 / --out-bands 4) and MODIS 11 um conversion for the rule.
(1) option off: bit-identical to the code before the option existed (golden values recorded from the old code);
(2) option on: 11 um TARGET values cannot reach any loss term; 11 um INPUT values do change the output; shapes, parameter count;
(3) the 4 um-only loss of a two-band prediction (11 um band weight 0) is the same number as the out4 loss on the 4 um slice;
(4) the physical 11 um path: Planck conversion, masks carried, no VIIRS value used."""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from golden_out4 import compute, fixed_batch  # noqa: E402

from nightrad.raw import losses, planck
from nightrad.raw.c11 import modis11_on_viirs_grid
from nightrad.raw.core import Prep, RunConfig, forward_loss, loss_from_scenes, run_scenes
from nightrad.raw.losses import LossConfig
from nightrad.raw.net import NetConfig, RawHarmonizer
from nightrad.raw.schema import split_batch

GOLD = json.load(open(os.path.join(os.path.dirname(__file__), "golden_out4_off.json")))
CPU = torch.device("cpu")


def _cfg(mode, out4):
    return RunConfig(net=NetConfig(width=8, latent=4, out4=out4), loss=LossConfig(mode=mode))


def _inp():
    return split_batch(fixed_batch(), CPU)[0]


@pytest.mark.parametrize("mode", ["legacy", "gamma"])
def test_option_off_is_bit_identical_to_before(mode):
    got, ref = compute(mode), GOLD[mode]
    same_build = got["torch"] == ref["torch"]
    assert got["n_params"] == ref["n_params"]
    assert got["B_hash"] == ref["B_hash"] if same_build else True
    assert got["E_hash"] == ref["E_hash"] if same_build else True
    for k in ("loss", "loss_after_step"):
        assert got[k] == ref[k] if same_build else abs(got[k] - ref[k]) <= 1e-5 * abs(ref[k]), (k, got[k], ref[k])
    for k, v in ref["logs"].items():
        assert got["logs"][k] == v if same_build else abs(got["logs"][k] - v) <= 1e-4 * max(1.0, abs(v)), k


def test_shapes_and_parameter_count():
    torch.manual_seed(0); m2 = RawHarmonizer(_cfg("legacy", False).net)
    torch.manual_seed(0); m1 = RawHarmonizer(_cfg("legacy", True).net)
    n2, n1 = sum(p.numel() for p in m2.parameters()), sum(p.numel() for p in m1.parameters())
    assert m1.decoder.head.conv.weight.shape[0] == 2 and m2.decoder.head.conv.weight.shape[0] == 4  # 1 background + 1 excess channel instead of 2 + 2
    assert n1 < n2
    print("parameters: both bands", n2, " 4 um only", n1, " difference", n2 - n1)
    inp = _inp(); cfg = _cfg("legacy", True)
    sc, _ = run_scenes(m1, Prep(inp, cfg), cfg, torch.Generator().manual_seed(1))
    for v in sc.values():
        assert v["B"].shape[1] == 1 and v["E"].shape[1] == 1 and (v["E"] >= 0).all()
    # shared parts start from the same weights as the two-band model (same seed): only the decoder head differs
    assert torch.equal(m1.stems["modis"].c1.conv.weight, m2.stems["modis"].c1.conv.weight)


@pytest.mark.parametrize("mode", ["legacy", "gamma"])
def test_11um_targets_cannot_influence_the_loss(mode):
    torch.manual_seed(0)
    inp = _inp(); cfg = _cfg(mode, True); model = RawHarmonizer(cfg.net)
    P = Prep(inp, cfg)
    with torch.no_grad():
        scenes, hidden = run_scenes(model, P, cfg, torch.Generator().manual_seed(3))
    base = float(loss_from_scenes(P, scenes, hidden, inp, cfg)[0])
    g = torch.Generator().manual_seed(9)
    for s in ("viirs", "modis"):  # scramble every 11 um TARGET value (log radiance and its local median) of both sensors
        P.logL[s][:, 1] = P.logL[s][:, 1] + 3.0 * torch.randn(P.logL[s][:, 1].shape, generator=g)
        P.med[s][:, 1] = P.med[s][:, 1] + 3.0 * torch.randn(P.med[s][:, 1].shape, generator=g)
    pert = float(loss_from_scenes(P, scenes, hidden, inp, cfg)[0])
    assert pert == base, (base, pert)
    # control: the same perturbation DOES change the two-band loss
    cfg2 = _cfg(mode, False); m2 = RawHarmonizer(cfg2.net); P2 = Prep(inp, cfg2)
    with torch.no_grad():
        sc2, h2 = run_scenes(m2, P2, cfg2, torch.Generator().manual_seed(3))
    b2 = float(loss_from_scenes(P2, sc2, h2, inp, cfg2)[0])
    for s in ("viirs", "modis"):
        P2.logL[s][:, 1] = P2.logL[s][:, 1] + 3.0 * torch.randn(P2.logL[s][:, 1].shape, generator=g)
    assert float(loss_from_scenes(P2, sc2, h2, inp, cfg2)[0]) != b2


def test_11um_inputs_still_change_the_output():
    torch.manual_seed(0)
    cfg = _cfg("legacy", True); model = RawHarmonizer(cfg.net)
    with torch.no_grad():  # make the head non-trivial so the output depends on the input
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    inp = _inp(); ref, _ = run_scenes(model, Prep(inp, cfg), cfg, None)
    for s in ("modis", "viirs"):
        inp2 = dict(inp); L = inp[f"{s}_L"].clone(); L[:, 1] = L[:, 1] * 1.3; inp2[f"{s}_L"] = L
        out, _ = run_scenes(model, Prep(inp2, cfg), cfg, None)
        k = (s, "clean")
        assert not torch.equal(out[k]["B"], ref[k]["B"]) or not torch.equal(out[k]["E"], ref[k]["E"]), s


def test_4um_only_loss_equals_two_band_loss_with_11um_weight_zero():
    """The reference's 4 um-only validation loss (band weights 1, 0) is the number the out4 loss computes on the same 4 um values."""
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(2, 2, 12, 12, generator=g) * 0.3; tgt = torch.randn(2, 2, 12, 12, generator=g) * 0.3
    mask = torch.rand(2, 12, 12, generator=g) > 0.2; w = torch.ones(2, 12, 12); sat = torch.rand(2, 12, 12, generator=g) > 0.9; rho = torch.randn(2, 12, 12, generator=g)
    c1 = LossConfig(mode="legacy", band_weights=(1.0, 1.0)); c0 = LossConfig(mode="legacy", band_weights=(1.0, 0.0))
    a = losses.term_loss(pred, tgt, mask, w, sat, c0, rho)[0]
    b = losses.term_loss(pred[:, :1], tgt[:, :1], mask, w, sat, c1, rho)[0]
    assert torch.allclose(a, b, rtol=0, atol=1e-7)
    mu, y = torch.rand(2, 2, 12, 12, generator=g) + 0.5, torch.rand(2, 2, 12, 12, generator=g) + 0.5
    ref = torch.ones(2)
    ga = losses.gamma_term(mu, y, mask, w, sat, LossConfig(mode="gamma", band_weights=(1.0, 0.0)), ref, y)
    gb = losses.gamma_term(mu[:, :1], y[:, :1], mask, w, sat, LossConfig(mode="gamma"), ref[:1], y[:, :1])
    assert torch.allclose(ga, gb, rtol=0, atol=1e-7)


def test_out4_rejects_unsupported_modes():
    with pytest.raises(AssertionError):
        RunConfig(net=NetConfig(out4=True), alloc=True)
    with pytest.raises(AssertionError):
        RunConfig(net=NetConfig(out4=True), mv_mode="hurdle")


def test_converted_modis_11um_physics_masks_and_independence():
    inp = _inp()
    L11, valid = modis11_on_viirs_grid(inp)
    common = (inp["modis_code"] == 0) & (inp["viirs_code"] == 0)
    assert torch.equal(valid, common) and valid.any()
    assert (L11[~valid] == 0).all()
    # physics: a blackbody of temperature T in the MODIS 11 um band becomes B(T) in the VIIRS I5 band
    T = torch.tensor([260.0, 290.0, 320.0, 360.0])
    Lm = planck.planck_radiance(T, planck.LAMBDA["modis_11"]).float()
    fake = {k: v.clone() for k, v in inp.items()}
    fake["modis_L"] = fake["modis_L"].clone(); fake["modis_L"][:4, 1] = Lm[:, None, None]
    fake["modis_code"] = torch.zeros_like(fake["modis_code"]); fake["viirs_code"] = torch.zeros_like(fake["viirs_code"])
    out, v = modis11_on_viirs_grid(fake)
    want = planck.planck_radiance(T, planck.LAMBDA["viirs_11"]).float()
    assert torch.allclose(out[:4, 5, 5], want, rtol=1e-4)
    # constant over a MODIS pixel (it carries its pixel's value), invalid cells never converted (NaN in them is harmless)
    pid = inp["modis_pid"][0]; b = 0
    for p in pid.unique()[:20]:
        cells = (pid == p) & valid[b]
        if cells.sum() > 1:
            assert torch.allclose(L11[b][cells], L11[b][cells][0].expand(int(cells.sum())), rtol=1e-6)
    nanned = dict(inp); nanned["modis_L"] = inp["modis_L"].clone(); nanned["modis_L"][:, 1][~valid] = float("nan")
    L2, _ = modis11_on_viirs_grid(nanned)
    assert torch.equal(L2, L11)
    # no VIIRS value is used: scrambling every VIIRS radiance leaves the result unchanged
    scr = dict(inp); scr["viirs_L"] = torch.randn_like(inp["viirs_L"]).abs() + 1.0
    L3, _ = modis11_on_viirs_grid(scr)
    assert torch.equal(L3, L11)


def test_scorer_runs_with_the_option_and_rule_variants():
    from nightrad.raw import protocol_eval as PE
    from nightrad.raw.synth import SyntheticTiles
    ds = SyntheticTiles(6, 0, 96, 96, cache=True)
    items = [ds[i] for i in range(6)]; tiles = {k: torch.stack([it[k] for it in items]) for k in items[0]}
    tiles["eval_pair_id"] = torch.tensor([0, 0, 1, 1, 2, 2]).reshape(6, 1, 1)  # the synthetic tiles carry no pair ids: three scene pairs of two tiles
    for out4 in (True, False):
        cfg = _cfg("legacy", out4); torch.manual_seed(0); model = RawHarmonizer(cfg.net)
        res = PE.collect(model, tiles, cfg, CPU, 0.0, bs=3)
        s = PE.summarize(res)
        assert ("model_c11" in s["rule"]) == (not out4), s["rule"].keys()
        assert "model" in s["rule"] and s["rule"]["model"]["flagged_cells_meas"] is not None
