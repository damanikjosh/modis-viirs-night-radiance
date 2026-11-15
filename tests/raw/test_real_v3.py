"""Real v3 tiles (skipped when the 50-tile sample is not on disk). Set FH_V3_SAMPLE=<dir with the extracted .npz files>."""
import glob
import os

import numpy as np
import pytest
import torch

from nightrad.raw.core import Prep, RunConfig
from nightrad.raw.data import NpzTiles
from nightrad.raw.schema import INPUT_KEYS, split_batch

D = os.environ.get("FH_V3_SAMPLE", "")
FILES = sorted(glob.glob(os.path.join(D, "*.npz"))) if D else []
pytestmark = pytest.mark.skipif(not FILES, reason="no real v3 sample")


def _non_test():
    out = []
    for f in FILES:
        with np.load(f, allow_pickle=False) as z:
            if str(z["split"]) != "test":
                out.append(f)
    return out


def test_adapter_contract_and_no_eval_in_inputs():
    ds = NpzTiles(_non_test()[:6])
    b = {k: torch.stack([ds[i][k] for i in range(len(ds))]) for k in ds[0]}
    inp, ev = split_batch(b)
    assert set(INPUT_KEYS) <= set(inp) and not any(k.startswith("eval_") for k in inp)
    assert all(k.startswith("eval_") for k in ev) and "eval_modis_cls" in ev and "eval_pair_id" in ev
    for s in ("modis", "viirs"):
        assert inp[f"{s}_L"].shape[1] == 2 and inp[f"{s}_code"].max() <= 5
    P = Prep(inp, RunConfig(daynight="both", min_valid_frac=0.0))
    # common mask = clear land in both; every masked-in cell has finite positive radiance
    for s in ("modis", "viirs"):
        L = inp[f"{s}_L"].float()
        assert (L.permute(0, 2, 3, 1)[P.common] > 0).all()
    z = np.load(_non_test()[0], allow_pickle=False)
    want = (z["modis_vclass"] == 2) & (z["viirs_vclass"] == 2)
    assert np.array_equal(P.common[0].numpy(), want) or len(_non_test()) == 0


def test_exclusion_rule():
    from nightrad.raw import exclude
    f = lambda la, lo, y, m, d: bool(exclude.excluded(la, lo, y, m, d)[0])
    assert f(-1.0, 113.0, 2015, 9, 15) and f(-1.0, 113.0, 2015, 9, 10) and f(-1.0, 113.0, 2015, 9, 27)
    assert f(1.4, 113.0, 2015, 9, 15) and not f(1.7, 113.0, 2015, 9, 15)      # 0.4 deg ~ 44 km inside the buffer; 0.7 deg ~ 78 km outside
    assert not f(-1.0, 113.0, 2015, 9, 9) and not f(-1.0, 113.0, 2019, 9, 15) and not f(-1.0, 100.0, 2015, 9, 15)


def _batch(n=6, files=None):
    ds = NpzTiles(files or _non_test()[:n])
    return {k: torch.stack([ds[i][k] for i in range(len(ds))]) for k in ds[0]}


def test_f1_invented_viirs_values_never_reach_inputs_or_loss():
    """Real tiles: cells with vclass != 2 (and VIIRS cells at the 367 K placeholder) hold zeros after the adapter; garbage injected there changes nothing."""
    from nightrad.raw.core import forward_loss
    from nightrad.raw.net import NetConfig, RawHarmonizer
    b = _batch(8)
    inp, _ = split_batch(b)
    for s in ("modis", "viirs"):
        bad = inp[f"{s}_code"] != 0
        assert bad.any() and (inp[f"{s}_L"].permute(0, 2, 3, 1)[bad] == 0).all() and (inp[f"{s}_sat"][bad] == 0).all()
    assert (inp["viirs_sat"] == 0).all()
    cfg = RunConfig(daynight="both", min_valid_frac=0.0); cfg.hide.enabled = False
    torch.manual_seed(0); net = RawHarmonizer(cfg.net).eval()
    l0, _ = forward_loss(net, inp, cfg)
    g = {k: v.clone() for k, v in inp.items()}
    for s in ("modis", "viirs"):
        bad = (g[f"{s}_code"] != 0)[:, None].expand_as(g[f"{s}_L"])
        g[f"{s}_L"] = torch.where(bad, torch.full_like(g[f"{s}_L"], 4.5623), g[f"{s}_L"])  # the 367 K placeholder everywhere outside the mask
        g[f"{s}_sat"] = torch.where(g[f"{s}_code"] != 0, torch.ones_like(g[f"{s}_sat"]), g[f"{s}_sat"])
    l1, _ = forward_loss(net, g, cfg)
    assert torch.equal(l0, l1)


def test_f1_placeholder_cells_removed_from_viirs_mask():
    n = 0
    for f in _non_test()[:60]:
        with np.load(f, allow_pickle=False) as z:
            at = np.isclose(z["viirs_rad4"], 4.562311667043153, rtol=1e-4)
            n += int(at.sum())
        t = NpzTiles([f])[0]
        assert (t["viirs_code"].numpy()[at] != 0).all()
    assert n > 0


def test_f2_no_input_channel_is_a_function_of_the_fire_product():
    """On valid cells no binary input channel may reproduce the product fire class: viirs_sat (stored) does -> adapter zeroes it; modis band21 does not."""
    n_stored = n_stored_fire = 0
    for f in _non_test():
        with np.load(f, allow_pickle=False) as z:
            v = z["viirs_vclass"] == 2
            n_stored += int(((z["viirs_sat"] > 0) & v).sum()); n_stored_fire += int(((z["viirs_sat"] > 0) & v & (z["eval_viirs_cls"] >= 4)).sum())
    assert n_stored == n_stored_fire  # documents F2: every stored VIIRS sat flag on a valid cell is a product fire pixel
    b = _batch(15, _non_test())
    inp, ev = split_batch(b)
    for s in ("modis", "viirs"):
        valid = inp[f"{s}_code"] == 0
        fire = ev[f"eval_{s}_cls"] >= 4
        for ch in ("sat", "band21"):
            c = inp[f"{s}_{ch}"] > 0
            if c.any():
                # the channel must be neither identical to the fire class nor contained in it on valid cells
                assert not ((c & valid) == (fire & valid)).all()
                assert (c & valid & ~fire).any() or not (c & valid).any(), f"{s}_{ch} is a subset of the product fire pixels"
    assert (inp["viirs_sat"] == 0).all()
