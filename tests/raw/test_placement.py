"""Placement probe: share loss, heuristics and hit-rate metrics on a hand-made 2x2-pixel tile (8x8 cells, 4x4 cells per pixel).
Written BEFORE src/nightrad/raw/placement.py."""
import numpy as np
import torch

from nightrad.raw import placement as PL
from nightrad.raw.mask import Footprints


def make_fp():
    H = W = 8; B = 1
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    blk = (yy // 4) * 2 + (xx // 4)                    # 0..3
    pid = (blk // 2) * 4 * W + (blk % 2) * 4           # dense-ish id = flat index of the block's first cell
    ids = {0: pid[0, 0].item(), 1: pid[0, 4].item(), 2: pid[4, 0].item(), 3: pid[4, 4].item()}
    # neighbour table order: scan -, scan +, track -, track + ; none = H*W
    none = H * W
    nb = {0: (none, ids[1], none, ids[2]), 1: (ids[0], none, none, ids[3]), 2: (none, ids[3], ids[0], none), 3: (ids[2], none, ids[1], none)}
    nbr = torch.tensor([[[nb[int(blk[y, x])][k] for x in range(W)] for y in range(H)] for k in range(4)])[None]
    fp = Footprints(pid[None].long(), nbr.long())
    fp.pix_ok = torch.ones(B, fp.n, dtype=torch.bool); fp.pix_ok[:, -1] = False
    return fp, blk


def test_centre_score_peaks_at_centre():
    fp, blk = make_fp()
    s = PL.centre_score(fp)[0]
    block0 = s[:4, :4]
    # four central-most cells of the 4x4 block tie for the max (centroid at 1.5,1.5): (1,1),(1,2),(2,1),(2,2)
    best = (block0 == block0.max()).nonzero().tolist()
    assert sorted(best) == [[1, 1], [1, 2], [2, 1], [2, 2]]


def test_psf_score_leans_toward_bright_neighbour():
    fp, blk = make_fp()
    exc = torch.zeros(1, 8, 8); exc[0, :4, 4:] = 5.0          # only pixel 1 (right of pixel 0) is bright
    s = PL.psf_score(fp, exc, own=False)[0]
    b0 = s[:4, :4]
    assert b0[:, 3].mean() > b0[:, 0].mean()                  # cells of pixel 0 nearer pixel 1 score higher
    assert float(b0[:, 0].max()) == 0.0 or b0[:, 0].max() < b0[:, 3].min()


def test_share_loss_learns_the_target_share():
    fp, blk = make_fp()
    e = torch.zeros(1, 8, 8); e[0, 1, 1] = 3.0; e[0, 1, 2] = 1.0   # pixel 0: 75 % / 25 % on two cells
    selc = (blk == 0)[None]
    lg = torch.zeros(1, 8, 8, requires_grad=True)
    l0 = float(PL.share_loss(lg, fp, e, selc))
    opt = torch.optim.SGD([lg], lr=2.0)
    for _ in range(200):
        opt.zero_grad(); PL.share_loss(lg, fp, e, selc).backward(); opt.step()
    p = torch.exp(lg - fp.logsumexp(lg))[0, :4, :4]
    assert abs(float(p[1, 1]) - 0.75) < 0.02 and abs(float(p[1, 2]) - 0.25) < 0.02
    assert float(PL.share_loss(lg, fp, e, selc)) < l0


def test_metrics_hit_and_shares():
    fp, blk = make_fp()
    e = torch.zeros(1, 8, 8); e[0, 1, 1] = 3.0; e[0, 1, 2] = 1.0
    selc = (blk == 0)[None]
    perfect = e.clone()
    r = PL.footprint_metrics(perfect, fp, e, selc, torch.tensor([7]), seed=0)
    assert r["hit"].tolist() == [1.0] and abs(r["top1"][0] - 0.75) < 1e-6 and abs(r["top3"][0] - 1.0) < 1e-6 and r["n_cells"][0] == 16 and r["pair"][0] == 7
    assert abs(r["max_share"][0] - 0.75) < 1e-6
    wrong = torch.zeros(1, 8, 8); wrong[0, 3, 3] = 1.0
    r2 = PL.footprint_metrics(wrong, fp, e, selc, torch.tensor([7]), seed=0)
    assert r2["hit"].tolist() == [0.0] and r2["top1"][0] == 0.0
    # constant score: random tie break -> hit rate ~ 1/16 over seeds
    h = [PL.footprint_metrics(torch.zeros(1, 8, 8), fp, e, selc, torch.tensor([7]), seed=s)["hit"][0] for s in range(400)]
    assert 0.02 < np.mean(h) < 0.12
