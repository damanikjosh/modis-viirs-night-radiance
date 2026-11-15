"""Mask-leak proof: nothing outside the common mask can influence anything inside the loss; exact equality, values AND gradients."""
import copy
import inspect

import pytest
import torch

from nightrad.raw.core import RunConfig, forward_loss, run_scenes, Prep
from nightrad.raw.hide import HideConfig
from nightrad.raw.layers import MConv, MGroupNorm, MaskCtx
from nightrad.raw.net import NetConfig, RawHarmonizer
from nightrad.raw.schema import EVAL_PREFIX, INPUT_KEYS, split_batch
from nightrad.raw.views import ViewConfig
from raw_helpers import tiny_cfg


def _perturb_outside(inp, common, garbage):
    out = {k: v.clone() for k, v in inp.items()}
    m = ~common
    for s in ("modis", "viirs"):
        L = out[f"{s}_L"]
        L[:, 0][m] = garbage; L[:, 1][m] = -garbage
        for k in ("vza", "cossza"):
            out[f"{s}_{k}"][m] = garbage
        out[f"{s}_psize"][:, 0][m] = garbage
        out[f"{s}_sat"][m] = 1; out[f"{s}_band21"][m] = 1
    return out


@pytest.mark.parametrize("variant", ["default", "tri", "hurdle", "conv3", "noskip", "lowres_sun_balance"])
@pytest.mark.parametrize("garbage", [1e6, float("nan"), float("inf")])
def test_masked_out_values_cannot_change_loss_or_gradients(small_batch, variant, garbage):
    inp, _ = small_batch
    kw = {}
    if variant == "tri": kw["view"] = ViewConfig(kernel="tri")
    if variant == "hurdle": kw["mv_mode"] = "hurdle"
    if variant == "conv3": kw["net"] = NetConfig(width=8, latent=4, arch="conv3")
    if variant == "noskip": kw["net"] = NetConfig(width=8, latent=4, arch="unet_nobottleneck_skip")
    if variant == "lowres_sun_balance":
        from nightrad.raw.losses import LossConfig
        kw["net"] = NetConfig(width=8, latent=4, delta_lowres=4, sun_term=True); kw["loss"] = LossConfig(class_balance=True)
    cfg = tiny_cfg(hide=HideConfig(p_tile=1.0), **kw)
    torch.manual_seed(0)
    model = RawHarmonizer(cfg.net)
    P = Prep(inp, cfg)
    res = []
    for data in (inp, _perturb_outside(inp, P.common, garbage)):
        model.zero_grad()
        loss, logs = forward_loss(model, data, cfg, torch.Generator().manual_seed(5))
        loss.backward()
        res.append((loss.detach().clone(), [p.grad.clone() for p in model.parameters() if p.grad is not None], {k: v for k, v in logs.items()}))
    assert torch.isfinite(res[0][0])
    assert torch.equal(res[0][0], res[1][0]), "loss changed when masked-out inputs changed"
    for g0, g1 in zip(res[0][1], res[1][1]):
        assert torch.equal(g0, g1)
    for k in res[0][2]:
        assert torch.equal(res[0][2][k], res[1][2][k]), k


def test_hidden_cells_are_scored_but_zeroed_on_input(small_batch):
    inp, _ = small_batch
    cfg = tiny_cfg(hide=HideConfig(p_tile=1.0, n_top=3))
    model = RawHarmonizer(cfg.net).eval()
    P = Prep(inp, cfg)
    sc0, hidden = run_scenes(model, P, cfg, torch.Generator().manual_seed(2))
    assert hidden["viirs"].any() and (hidden["viirs"] <= P.common).all()
    inp2 = {k: v.clone() for k, v in inp.items()}
    inp2["viirs_L"][:, :][hidden["viirs"][:, None].expand(-1, 2, -1, -1)] = 123.0  # change only the hidden cells (inside the mask!)
    P2 = Prep(inp2, cfg)
    sc1, _ = run_scenes(model, P2, cfg, hidden=hidden)
    # hidden pass of the VIIRS-sourced scene cannot see hidden values ...
    assert torch.equal(sc0[("viirs", "hid")]["E"], sc1[("viirs", "hid")]["E"]) and torch.equal(sc0[("viirs", "hid")]["B"], sc1[("viirs", "hid")]["B"])
    # ... but the loss target does (they are scored)
    l0 = forward_loss(model, inp, cfg, torch.Generator().manual_seed(2))[1]["vv/loss"]
    l1 = forward_loss(model, inp2, cfg, torch.Generator().manual_seed(2))[1]["vv/loss"]
    assert not torch.equal(l0, l1)


def test_nonoverlap_cells_never_in_loss_and_input_mask_equals_loss_mask(small_batch):
    inp, _ = small_batch
    cfg = tiny_cfg()
    P = Prep(inp, cfg)
    one_only = ((inp["viirs_code"] == 0) ^ (inp["modis_code"] == 0))
    assert one_only.any()
    assert not (P.common & one_only).any()
    assert (P.ok_foot <= P.common).all()
    # the network domain mask passed to forward IS the loss mask (common)
    seen = []
    model = RawHarmonizer(cfg.net)
    orig = model.forward
    model.forward = lambda x, m, *a, **k: (seen.append(m.clone()), orig(x, m, *a, **k))[1]
    run_scenes(model, P, cfg, torch.Generator().manual_seed(0))
    assert seen and all(torch.equal(seen[0][i * 8:(i + 1) * 8], P.common) for i in range(len(seen[0]) // 8))


def test_partial_conv_has_no_edge_signature_and_layers_keep_the_zero_invariant():
    """Layers assume (and preserve) exact zeros outside the mask; the network-level tests above prove no outside value can enter."""
    torch.manual_seed(0)
    m = torch.zeros(1, 1, 12, 12); m[:, :, 2:9, 3:11] = 1; m[:, :, 5, 5] = 0
    ctx = MaskCtx(m)
    conv = MConv(1, 1, 3, 1)
    with torch.no_grad():
        conv.conv.weight.fill_(1.0 / 9)  # box filter
    x = m.clone()  # ones inside the mask, 0 outside
    y = conv(x, ctx)
    assert torch.allclose(y[m.bool()], torch.ones(int(m.sum())), atol=1e-6)  # constant stays constant to the mask edge (zero padding would give < 1)
    assert (y[~m.bool()] == 0).all()
    gn = MGroupNorm(8)
    xx = torch.randn(1, 8, 12, 12) * m
    yy = gn(xx, ctx)
    assert (yy[(m == 0).expand_as(yy)] == 0).all()
    inside = yy[m.expand_as(yy).bool()].view(8, -1)  # normalised over in-mask cells only: zero mean, unit variance per group
    assert torch.allclose(inside.mean(1), torch.zeros(8), atol=1e-5) and torch.allclose(inside.var(1, unbiased=False), torch.ones(8), atol=1e-3)


def test_forward_cannot_receive_eval_arrays(small_batch):
    sig = inspect.signature(RawHarmonizer.forward)
    assert not any(p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in sig.parameters.values())
    assert not any(n.startswith(EVAL_PREFIX) or "eval" in n for n in sig.parameters)
    assert not any(k.startswith(EVAL_PREFIX) for k in INPUT_KEYS)
    for banned in ("dt", "time", "lat", "lon", "date", "land", "cover", "fire", "frp", "conf"):
        assert not any(banned in n for n in sig.parameters), banned
    # eval_* in a batch are stripped by split_batch and never change the model output
    inp, ev = small_batch
    assert ev and all(k.startswith(EVAL_PREFIX) for k in ev) and not any(k.startswith(EVAL_PREFIX) for k in inp)
    cfg = tiny_cfg()
    model = RawHarmonizer(cfg.net)
    torch.manual_seed(0)
    b = {**inp, **{k: torch.full_like(v, float("nan") if v.is_floating_point() else 7) for k, v in ev.items()}}
    inp_b, _ = split_batch(b)
    a_ = forward_loss(model, inp, cfg, torch.Generator().manual_seed(0))[0]
    b_ = forward_loss(model, inp_b, cfg, torch.Generator().manual_seed(0))[0]
    assert torch.equal(a_, b_)


@pytest.mark.parametrize("kw", [dict(), dict(mv_mode="hurdle"), dict(mv_mode="measured"), dict(alloc=True), dict(tie11=True), dict(daynight="night"),
                                dict(net=NetConfig(width=8, latent=4, arch="conv3")), dict(net=NetConfig(width=8, latent=4, arch="unet_nobottleneck_skip")),
                                dict(net=NetConfig(width=8, latent=4, delta_lowres=4, delta_max=0.1, sun_term=True)),
                                dict(hide=HideConfig(enabled=False)), dict(hide=HideConfig(partial_p=1.0))])
def test_every_option_runs_backward_with_finite_grads(small_batch, kw):
    inp, _ = small_batch
    cfg = tiny_cfg(**kw)
    model = RawHarmonizer(cfg.net)
    loss, logs = forward_loss(model, inp, cfg, torch.Generator().manual_seed(0))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for t in ("mm", "vv", "vm", "mv"):
        assert f"{t}/loss" in logs and f"{t}/share_fire" in logs  # per-term, per-class logging from step 0
