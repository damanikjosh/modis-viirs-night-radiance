"""Review tests of model and training. A passing test verifies a property; a test marked xfail(strict) reproduces a documented finding and must keep failing:
the Group-share loss paints about 6x the measured energy when the fire position inside a pixel is unknown (paper: loss-family ablation), and the hiding windows are centred
on the brightest cells by design (jitter 3), so hole position is not independent of fire position."""
import glob, math, os
from argparse import Namespace
import numpy as np, pytest, torch
from nightrad.raw.core import Prep, RunConfig, run_scenes
from nightrad.raw.hide import HideConfig, hide_regions
from nightrad.raw.losses import LossConfig, term_loss
from nightrad.raw.net import NetConfig, RawHarmonizer
from raw_helpers import tiny_cfg

# ---- verified correct: the target sensor's measured values never reach the path that predicts it
@pytest.mark.parametrize("src,other", [("viirs", "modis"), ("modis", "viirs")])
def test_cross_sensor_scene_does_not_depend_on_target_sensor_values(small_batch, src, other):
    inp, _ = small_batch
    cfg = tiny_cfg(hide=HideConfig(p_tile=1.0)); torch.manual_seed(0)
    model = RawHarmonizer(cfg.net).eval()
    P = Prep(inp, cfg)
    sc0, h = run_scenes(model, P, cfg, torch.Generator().manual_seed(1))
    g = {k: v.clone() for k, v in inp.items()}
    g[f"{other}_L"] = torch.where(P.common[:, None], g[f"{other}_L"] * 7.0 + 3.0, g[f"{other}_L"])  # change the other sensor INSIDE the mask
    P2 = Prep(g, cfg)
    sc1, _ = run_scenes(model, P2, cfg, hidden=h)
    for v in ("clean",):
        assert torch.equal(sc0[(src, v)]["E"], sc1[(src, v)]["E"]) and torch.equal(sc0[(src, v)]["B"], sc1[(src, v)]["B"])

# ---- FINDING H1: the hole left by hiding is centred on the brightest cell, and the mask channel shows it
@pytest.mark.xfail(strict=True, reason="documented finding: hiding windows are centred on the brightest fire-level cells by design")
def test_hole_position_is_independent_of_fire_position():
    from nightrad.raw.synth import SyntheticTiles
    from torch.utils.data import DataLoader
    from nightrad.raw.schema import split_batch
    inp, _ = split_batch(next(iter(DataLoader(SyntheticTiles(40, 5, 48, 48), batch_size=40))))
    P = Prep(inp, tiny_cfg())
    cfg = HideConfig(p_tile=1.0, n_top=1, n_random=0)
    hid = hide_regions(P.rho4["viirs"], P.common, cfg, torch.Generator().manual_seed(0))
    n = hit = 0
    for b in range(40):
        ok = P.common[b] & (P.rho4["viirs"][b] >= 0.5)
        if not ok.any() or not hid[b].any(): continue
        top = torch.where(ok, P.rho4["viirs"][b], torch.tensor(-9.0)).flatten().argmax()
        ty, tx = divmod(int(top), 48); ys, xs = hid[b].nonzero().T
        n += 1; hit += max(abs(int(ys.float().mean()) - ty), abs(int(xs.float().mean()) - tx)) <= 2
    assert n > 5
    assert hit / n < 0.5, f"hole centre within 2 cells of the peak in {hit}/{n} tiles (should be ~chance)"

# ---- FINDING C1: with fixed class shares the best deterministic prediction inside a bright pixel is fire-level everywhere
def _tile_loss(c, pos, E, cfg, H=16, k=3):
    tgt = torch.zeros(1, 2, H, H); r0 = H // 2 - 1; py, px = r0 + pos // k, r0 + pos % k
    tgt[0, 0, py, px] = math.log(1 + E); rho = torch.zeros(1, H, H); rho[0, py, px] = math.log(1 + E)
    pred = tgt.clone(); pred[0, 0, r0:r0 + k, r0:r0 + k] = c
    l, _ = term_loss(pred, tgt, torch.ones(1, H, H, dtype=torch.bool), torch.ones(1, H, H), torch.zeros(1, H, H, dtype=torch.bool), cfg, rho)
    return float(l)

@pytest.mark.xfail(strict=True, reason="documented finding: the Group-share loss does not conserve energy with unknown fire position (the Gamma family does, see test_gamma_loss.py)")
@pytest.mark.parametrize("cfg", [LossConfig(), LossConfig(class_balance=False)])
def test_loss_optimum_with_unknown_fire_position_conserves_energy(cfg):
    E, k = 20.0, 3
    grid = np.linspace(0, math.log(1 + E), 200)
    tot = [np.mean([_tile_loss(c, p, E, cfg) for p in range(k * k)]) for c in grid]
    best = math.exp(grid[int(np.argmin(tot))]); cons = 1 + E / (k * k)
    assert 0.6 * cons <= best <= 1.6 * cons, f"optimal pixel-wide level {best:.2f} x background; energy-conserving = {cons:.2f}"

# ---- FINDING S1: a hole at the VIIRS 367 K ceiling (product-saturated fire) sits next to real fire (needs the real sample; non-test tiles only)
D = os.environ.get("FH_V3_SAMPLE", "")
@pytest.mark.skipif(not D, reason="needs FH_V3_SAMPLE")
def test_mask_holes_do_not_mark_fire_locations():
    import torch.nn.functional as F
    from nightrad.raw.data import NpzTiles
    from nightrad.raw.schema import split_batch
    near_f = near_n = els_f = els_n = 0
    for f in sorted(glob.glob(os.path.join(D, "*.npz"))):
        z = np.load(f)
        if str(z["split"]) == "test": continue
        hole = torch.from_numpy(np.isclose(z["viirs_rad4"], 4.562311667043153, rtol=1e-4))[None]
        if not hole.any(): continue
        t = NpzTiles([f])[0]; inp, _ = split_batch({k: v[None] for k, v in t.items()})
        P = Prep(inp, RunConfig(daynight="both", min_valid_frac=0.0))
        near = (F.max_pool2d(hole.float()[:, None], 3, 1, 1)[:, 0] > 0) & P.common; fire = P.common & (P.rho4["viirs"] >= 0.5)
        near_f += int((fire & near).sum()); near_n += int(near.sum()); els_f += int((fire & ~near).sum()); els_n += int((P.common & ~near).sum())
    assert near_n > 0 and (near_f / near_n) < 3 * (els_f / els_n), f"P(fire | next to a hole) = {near_f/near_n:.3f} vs {els_f/els_n:.3f} elsewhere"

# ---- FINDING T1: after a resume the sweep's best-checkpoint key restarts at [9, inf]
def test_resume_restores_best_key(tmp_path):
    import json
    from nightrad.raw.synth import SyntheticTiles
    from nightrad.raw.train import train
    class DS(SyntheticTiles):
        def __getitem__(self, i):
            t = dict(super().__getitem__(i)); t["eval_pair_id"] = torch.tensor(i % 6, dtype=torch.int32); return t
    halves = tmp_path / "h.json"; json.dump({"A": [0, 2, 4], "B": [1, 3, 5]}, open(halves, "w"))
    cfg = RunConfig(net=NetConfig(width=8, latent=4), daynight="both")
    mk = lambda ep, resume, stop=None: Namespace(out=str(tmp_path / "r"), epochs=ep, bs=8, lr=2e-3, wd=1e-4, warmup=2, amp="off", channels_last=False, compile=False,
        workers=0, seed=1, resume=resume, deterministic=False, clip=1.0, stop_after=stop, halves=str(halves), eval_every=1)
    tr, va = DS(24, 0, 48, 48, cache=True), DS(24, 10**6, 48, 48, cache=True)
    a1 = mk(4, False, stop=2); train(a1, tr, va, cfg, log=lambda *_: None)
    key_before = list(a1._best_key)
    a2 = mk(4, True); train(a2, tr, va, cfg, log=lambda *_: None)
    rows = [json.loads(l) for l in open(tmp_path / "r" / "log.jsonl")]
    first_after = [r for r in rows if r["epoch"] == 2][0]
    # a worse epoch must not displace the stored best: with the bug the first post-resume eval is always 'best'
    assert not (first_after["is_best"] and first_after["sweep"]["A"]["mv3_fire|model"]["median"] > key_before[1]), "resume reset the best key"
