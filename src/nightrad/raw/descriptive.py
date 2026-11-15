"""DESCRIPTIVE, POST-HOC measures that need new scoring code (never pass lines, never used to pick a model, not part of the one-time closed test).

Four measures, each defined on top of the objects the stored protocol already uses (same tiles, same masks, same fire rule, same simple methods; nothing new is thresholded):
  1. Fractions skill score (FSS, Roberts & Lean 2008) of FIRE CELLS at neighbourhoods of 1, 3, 5, 9, 15, 27 cells (375 m, 1.1 km, 1.9 km, 3.4 km, 5.6 km, 10 km; headline: 1 / 3 / 9).
  2. Precision-recall curve and average precision (AP) of fire at BLOCK level (3x3 MODIS pixels, the protocol's complete blocks, score = predicted block energy) and at CELL level (375 m cells,
     score = predicted cell excess), plus the critical success index (CSI) at the operating point of the documented flag rule.
  3. "Right hottest cell" rate inside MODIS pixels that hold VIIRS fire, with the chance level of a random pick.
  4. Amount skill at 3 km against an even spread, per scene pair (1 - model error / even-spread error).
All fire definitions are the documented ones: a VIIRS cell is FIRE when the documented flag rule (src/nightrad/raw/rules.py, VIIRS night) fires on MEASURED radiance; a method FLAGS a cell when the same rule fires on its
PREDICTED radiance (4 and 11 um). Blocks: protocol_eval_v2 (complete 3x3 blocks, one item per MODIS pixel, label = block holds >= 1 fire cell). Fire level = ln(L4 / local median) >= 0.5 (sweep_eval.FIRE).
Nothing here reads a REPORT tile: tile choice goes through select_tile_indices(), which only knows 'select' and 'oldval' and runs the same guard (nf2_split.check_scoring_set) as scripts/score_run.py."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from nightrad.raw import planck
from nightrad.raw.blocks import once_per_footprint
from nightrad.raw.hide import HideConfig  # noqa: F401  (ref_cfg builds one)
from nightrad.raw.rules import detect
from nightrad.raw.schema import split_batch
from nightrad.raw.sweep_eval import FIRE, ref_cfg
from nightrad.raw.views import ModisView, modis_to_viirs_background

METHODS = ("model", "model_off", "uniform", "copy_planck", "copy_raw", "excess_only", "const")
SIMPLE = ("uniform", "copy_planck", "copy_raw", "excess_only", "const")
SCALES = (1, 3, 5, 9, 15, 27)        # neighbourhood edge in 375 m cells (odd, centred): 0.375 / 1.1 / 1.9 / 3.4 / 5.6 / 10.1 km
HEADLINE_SCALES = (1, 3, 9)
HIST_LO, HIST_HI, HIST_PER_DECADE = 1e-6, 1e3, 1000   # cell-level precision-recall: log-spaced score bins (ratio 1.0023 between neighbours) + one bin for score < 1e-6
MIN_POS_BLOCKS, MIN_NEG_BLOCKS = 10, 100               # protocol_eval_v2: a scene pair qualifies with >= 10 fire blocks and >= 100 other blocks
MIN_PIXELS = 10                                        # hottest-cell rate: a scene pair needs >= 10 eligible MODIS pixels
MIN_ITEMS = 10                                         # amount skill: >= 10 MODIS-selected items per pair (amendment 4)
TIE_RTOL = 1e-6
AMOUNT_EPS = 1e-3
HOT_VARIANTS = ("flagged", "fire_level")


def hist_edges() -> torch.Tensor:
    n = int(round(np.log10(HIST_HI / HIST_LO) * HIST_PER_DECADE))
    return torch.tensor(np.logspace(np.log10(HIST_LO), np.log10(HIST_HI), n + 1), dtype=torch.float64)


NB = len(hist_edges()) + 1   # bins: 0 = score below HIST_LO (incl. 0), 1..n = log bins, n+1 = >= HIST_HI


# ---------------------------------------------------------------- guarded tile choice
def select_tile_indices(data: str, set_name: str, split_episodes: str = "configs/nf2_split_episodes.json", nf2_split: str = "configs/nf2_split.json",
                        allow_unnamed_reason: str | None = None, retired: str | None = None, split_override_reason: str | None = None):
    """Tile indices of the pack that may be scored, and the guard record. ONLY 'select' (nf2 SELECT episodes) and 'oldval' (the old 9-pair validation pack, ids not in nf2) exist here: a REPORT or
    final request cannot be expressed. Same selection as scripts/score_run.py for these sets, same guard call (nf2_split.check_scoring_set, which refuses REPORT and retired ids by tile id)."""
    from nightrad.raw.nf2_split import DEFAULT_RETIRED, check_scoring_set
    if set_name not in ("select", "oldval"):
        raise ValueError(f"set {set_name!r} is not allowed here: only 'select' and 'oldval'")
    ep = np.load(os.path.join(data, "eval_episode.npy"), mmap_mode="r") if os.path.exists(os.path.join(data, "eval_episode.npy")) else None
    n = json.load(open(os.path.join(data, "meta.json")))["n"]
    keep = np.ones(n, bool)
    if set_name == "select":
        ep_names = {v: k for k, v in json.load(open(os.path.join(data, "episodes.json"))).items()}
        first = np.asarray(ep).reshape(n, -1)[:, 0]
        keep &= np.isin(np.array([ep_names.get(int(e), "") for e in first]), json.load(open(split_episodes))["select"])
    idx = [int(i) for i in np.flatnonzero(keep)]
    guard = check_scoring_set(data, nf2_split, set_name, allow_unnamed_reason, False, indices=idx, split_override_reason=split_override_reason, retired_json=retired or DEFAULT_RETIRED)
    return idx, guard


def names_sha256(data: str, idx) -> str | None:
    f = os.path.join(data, "names.json")
    if not os.path.exists(f):
        return None
    names = json.load(open(f))
    return hashlib.sha256("\n".join(names[i] for i in idx).encode()).hexdigest()


# ---------------------------------------------------------------- measure 1: fractions skill score
def fss_terms(obs: torch.Tensor, fc: torch.Tensor, valid: torch.Tensor, k: int):
    """Per-tile sums of the fractions skill score for a k x k neighbourhood (k odd; 1 = the cell itself).
    obs, fc: [B,H,W] bool fire fields; valid: [B,H,W] bool domain. Fraction at a cell = fire cells / valid cells inside the k x k window centred on it (window cut at the tile edge and at invalid
    cells: only valid cells count; no fraction where the window holds no valid cell). The score is evaluated at valid centres only.
    -> (sum (Po - Pf)^2, sum (Po^2 + Pf^2), number of centres) as float64 [B]; FSS = 1 - sum1 / sum2 once pooled over a scene pair (0/0, i.e. no fire in either field anywhere: undefined, never counted)."""
    assert k % 2 == 1 and k >= 1, "neighbourhood size must be odd"
    v = valid.float()[:, None]
    den = F.avg_pool2d(v, k, 1, k // 2)
    fr = lambda x: (F.avg_pool2d((x & valid).float()[:, None] * v, k, 1, k // 2) / den.clamp(min=1e-12))[:, 0]
    po, pf = fr(obs), fr(fc)
    m = valid.double()
    return (((po - pf).double() ** 2) * m).sum((1, 2)), (((po.double() ** 2) + (pf.double() ** 2)) * m).sum((1, 2)), m.sum((1, 2))


def fss_value(num: float, den: float) -> float:
    return float("nan") if den <= 0 else float(1.0 - num / den)


# ---------------------------------------------------------------- measure 2: precision-recall, average precision
def curve_from_counts(pos: np.ndarray, neg: np.ndarray):
    """pos / neg: counts per score bin, ordered from LOW to HIGH score. Threshold = every bin boundary (ties share a bin, so a tie group is one step).
    -> precision, recall (descending threshold), average precision = sum (R_n - R_{n-1}) P_n (the step definition, no interpolation), base rate. NaN when there is no positive."""
    pos = np.asarray(pos, np.float64)[::-1]; neg = np.asarray(neg, np.float64)[::-1]
    P, N = pos.sum(), neg.sum()
    if P == 0:
        return np.array([]), np.array([]), float("nan"), float("nan")
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    seen = (pos + neg) > 0
    tp, fp = tp[seen], fp[seen]
    prec, rec = tp / (tp + fp), tp / P
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return prec, rec, ap, float(P / (P + N))


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """Exact AP of a score against a boolean label (tied scores form one threshold)."""
    score, label = np.asarray(score), np.asarray(label).astype(bool)
    u, inv = np.unique(score, return_inverse=True)
    return curve_from_counts(np.bincount(inv, weights=label, minlength=len(u)), np.bincount(inv, weights=~label, minlength=len(u)))[2]


def pr_curve(score: np.ndarray, label: np.ndarray):
    score, label = np.asarray(score), np.asarray(label).astype(bool)
    u, inv = np.unique(score, return_inverse=True)
    p, r, ap, base = curve_from_counts(np.bincount(inv, weights=label, minlength=len(u)), np.bincount(inv, weights=~label, minlength=len(u)))
    return p, r, ap, base


def thin_curve(prec: np.ndarray, rec: np.ndarray, n: int = 250):
    """Display thinning only (the AP is computed before): keep ~n points spread evenly in recall, always the first and the last."""
    if len(rec) <= n:
        return [float(x) for x in rec], [float(x) for x in prec]
    sel = np.unique(np.r_[0, np.searchsorted(rec, np.linspace(0, 1, n), side="left").clip(0, len(rec) - 1), len(rec) - 1])
    return [float(rec[i]) for i in sel], [float(prec[i]) for i in sel]


def csi_from_counts(tp: float, fp: float, fn: float) -> float:
    """Critical success index = hits / (hits + misses + false alarms). NaN when all three are zero."""
    d = tp + fp + fn
    return float("nan") if d == 0 else float(tp / d)


# ---------------------------------------------------------------- measure 3: right hottest cell
def hottest_cell(V: torch.Tensor, T: torch.Tensor, pid: torch.Tensor, n: int, h: int, w: int):
    """Per MODIS pixel: is the hottest cell of V (a method's predicted radiance) a truly hottest cell of T (measured radiance)?
    V, T: [B,H*W] float; pid [B,H*W] long (pixel id of every cell; id n-1 = none). Returns per pixel [B,n] float tensors:
      exact   = probability that the method's hottest cell is a truly hottest cell. Ties are handled symmetrically: the method picks uniformly among the cells tied at ITS maximum (a flat spread therefore
                gets exactly the chance level), and every cell tied at the MEASURED maximum counts as 'the hottest' (measured radiance is quantised, so ~20 % of pixels have a tied maximum);
                expected credit = |tied(V) ∩ tied(T)| / |tied(V)|,
      within1 = same, but the picked cell may also be a neighbour (Chebyshev distance <= 1 on the cell grid, inside the same pixel) of a truly hottest cell,
      chance_exact / chance_within1 = the same quantities for a method that picks one of the pixel's cells uniformly at random (|tied(T)| / cells; cells within one of a hottest cell / cells),
      truth_tied = the measured maximum is shared by more than one cell (reported, never dropped).
    Tolerance for 'tied': value >= max * (1 - 1e-6)."""
    b = V.shape[0]; dev = V.device
    amax = lambda x: torch.full((b, n), -1e30, dtype=x.dtype, device=dev).scatter_reduce(1, pid, x, "amax", include_self=True)
    tol = lambda mx: mx - mx.abs() * TIE_RTOL
    mxv, mxt = amax(V), amax(T)
    top_v = V >= torch.gather(tol(mxv), 1, pid); top_t = T >= torch.gather(tol(mxt), 1, pid)
    tt = top_t.view(b, 1, h, w).float(); pp = pid.view(b, 1, h, w)
    near = top_t.clone()   # cell is a hottest cell, or a neighbour (same pixel) of one
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sh_t = F.pad(tt, (1, 1, 1, 1))[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            sh_p = F.pad(pp, (1, 1, 1, 1), value=-1)[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            near |= ((sh_t > 0) & (sh_p == pp)).view(b, -1)
    zeros = lambda: torch.zeros(b, n, device=dev)
    cnt = zeros().scatter_add_(1, pid, torch.ones_like(V, dtype=torch.float32))
    t_v = zeros().scatter_add_(1, pid, top_v.float()).clamp(min=1.0)
    t_t = zeros().scatter_add_(1, pid, top_t.float())
    exact = zeros().scatter_add_(1, pid, (top_v & top_t).float()) / t_v
    within = zeros().scatter_add_(1, pid, (top_v & near).float()) / t_v
    ch_e = t_t / cnt.clamp(min=1.0); ch_w = zeros().scatter_add_(1, pid, near.float()) / cnt.clamp(min=1.0)
    tied = (t_t > 1.5).float()
    return exact, within, ch_e, ch_w, tied


# ---------------------------------------------------------------- measure 4: amount skill vs even spread
def amount_skill(items: dict, methods=METHODS, refs=("uniform", "excess_only"), min_items: int = MIN_ITEMS, n_boot: int = 4000, seed: int = 0) -> dict:
    """items: arrays 'pair', 'meas' and 'pred_<method>' (amount_eval.collect_items output at one scale). Error of a method in a scene pair = mean |ln((pred + eps) / (meas + eps))| over the pair's
    items (amendment 4 error, eps = 1e-3). Skill of method m against reference r in a pair = 1 - err_m / err_r (1 = perfect, 0 = as good as the reference, < 0 = worse). Pairs with fewer than `min_items`
    items are left out. Returns per method and reference: per-pair skill (all qualifying pairs), mean / median / quartiles / share of pairs above 0 / min / max, a pair-clustered 95 % range of the mean
    (4000 resamples, seed 0), and the item-weighted skill over all items pooled."""
    pair, meas = np.asarray(items["pair"]), np.asarray(items["meas"], float)
    u, inv = np.unique(pair, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(u))]
    qual = [i for i, g in enumerate(groups) if len(g) >= min_items]
    err = {}
    for m in methods:
        if f"pred_{m}" in items:
            e = np.abs(np.log((np.asarray(items[f"pred_{m}"], float) + AMOUNT_EPS) / (meas + AMOUNT_EPS)))
            err[m] = (np.array([e[groups[i]].mean() for i in qual]), float(e.mean()))
    rng = np.random.default_rng(seed)
    out = {"pairs": [int(u[i]) for i in qual], "n_items_per_pair": [int(len(groups[i])) for i in qual], "n_items": int(len(meas)), "error_per_pair": {m: [float(x) for x in v[0]] for m, v in err.items()},
           "error_pooled": {m: v[1] for m, v in err.items()}, "skill": {}}
    for r in refs:
        if r not in err:
            continue
        out["skill"][r] = {}
        for m in err:
            with np.errstate(divide="ignore", invalid="ignore"):
                s = 1.0 - err[m][0] / err[r][0]
            ok = np.isfinite(s); sv = s[ok]
            if len(sv) == 0:
                out["skill"][r][m] = {"n_pairs": 0}; continue
            bs = rng.choice(sv, (n_boot, len(sv))).mean(1) if len(sv) >= 2 else None   # fewer than 2 pairs: no range (None, never NaN)
            q = np.percentile(sv, [25, 50, 75])
            out["skill"][r][m] = {"per_pair": [None if not o else float(x) for x, o in zip(s, ok)], "n_pairs": int(len(sv)), "mean": float(sv.mean()), "median": float(q[1]), "q25": float(q[0]), "q75": float(q[2]),
                                  "min": float(sv.min()), "max": float(sv.max()), "share_pairs_above_0": float((sv > 0).mean()), "lo": None if bs is None else float(np.percentile(bs, 2.5)), "hi": None if bs is None else float(np.percentile(bs, 97.5)),
                                  "pooled": float(1.0 - err[m][1] / err[r][1]) if err[r][1] > 0 else None}
    return out


# ---------------------------------------------------------------- collection
class Acc:
    """Additive per-scene-pair accumulators; chunks of tiles can be added in any order (all sums, exact integers or float64)."""

    def __init__(self):
        self.d = defaultdict(lambda: None)
        self.blocks = defaultdict(lambda: ([], []))   # (method, pair) -> (scores, labels)
        self.n_tiles = 0

    def add(self, key, vec):
        v = np.asarray(vec, np.float64)
        self.d[key] = v if self.d[key] is None else self.d[key] + v

    def get(self, key, size=None):
        v = self.d.get(key)
        return v if v is not None else (np.zeros(size) if size else None)


def install_run_cache():
    """The collectors of one scoring pass see the same model output for the same tiles: run the network once per batch (same device as scripts/score_run.py does; identical numbers)."""
    import nightrad.raw.core as core
    if getattr(core.run_scenes, "_fh_cached", False):
        return
    orig, cache = core.run_scenes, {}

    def cached(model, P, cfg, gen=None, amp_dtype=None, hidden=None, mark=None):
        key = (id(model), P.inp["viirs_L"].shape[0], float(P.inp["viirs_L"].double().sum()), float(P.inp["modis_L"].double().sum()))
        if key not in cache:
            if len(cache) > 400:
                cache.clear()
            cache[key] = orig(model, P, cfg, gen, amp_dtype, hidden) if mark is None else orig(model, P, cfg, gen, amp_dtype, hidden, mark)
        return cache[key]
    cached._fh_cached = True; cached._cache = cache
    core.run_scenes = cached


@torch.no_grad()
def collect(model, tiles: dict, cfg_run, dev, const: float, acc: Acc, bs: int = 8, amp_dtype=None, methods=METHODS):
    """Adds the tiles to `acc`. model=None -> simple methods only (model and model_off dropped)."""
    import nightrad.raw.core as core
    from nightrad.raw.core import Prep
    methods = [m for m in methods if model is not None or m not in ("model", "model_off")]
    if model is not None:
        model.eval()
    cfg = ref_cfg(cfg_run); cfg.hide = replace(cfg.hide, enabled=False)
    mv = ModisView(cfg.view); f4 = torch.tensor([mv.f4, mv.f11], device=dev)[None, :, None, None]
    n = tiles["viirs_L"].shape[0]; pairs_all = tiles["eval_pair_id"].reshape(n, -1)[:, 0].numpy().astype(np.int64)
    edges = hist_edges().to(dev)
    day0_all = None
    for i0 in range(0, n, bs):
        idx = np.arange(i0, min(n, i0 + bs)); nb_ = len(idx)
        batch = {k: v[idx] for k, v in tiles.items()}
        inp, ev = split_batch(batch, dev); pt = pairs_all[idx]
        P = Prep(inp, cfg); common = P.common
        scm = None
        if model is not None:
            sc, _ = core.run_scenes(model, P, cfg, None, amp_dtype); scm = sc[("modis", "clean")]
        Lm = torch.exp(P.logL["modis"]); Bmed = torch.exp(P.med["modis"]); Bv = torch.exp(P.med["viirs"][:, 0]); tv = P.logL["viirs"][:, 0]
        pred = {"uniform": (modis_to_viirs_background(Bmed) + (Lm - Bmed).clamp(min=0) / f4)[:, 0], "copy_planck": modis_to_viirs_background(Lm)[:, 0], "copy_raw": Lm[:, 0],
                "excess_only": Bv + (Lm - Bmed).clamp(min=0)[:, 0] / f4[:, 0], "const": torch.full_like(Bv, float(np.exp(const)))}
        # 11 um twins: copied from protocol_eval.collect (same expressions) so the flag rule sees the same predicted view (cross-checked against protocol_eval in tests and against stored score files)
        pred11 = {"uniform": modis_to_viirs_background(Bmed)[:, 1] + (Lm - Bmed).clamp(min=0)[:, 1] / f4[:, 1, 0, 0], "copy_planck": modis_to_viirs_background(Lm)[:, 1], "copy_raw": Lm[:, 1],
                  "excess_only": torch.exp(P.med["viirs"][:, 1]) + (Lm - Bmed).clamp(min=0)[:, 1] / f4[:, 1, 0, 0], "const": torch.full_like(Bv, float(np.exp(const)))}
        if scm is not None:
            pred["model"], pred["model_off"] = (scm["B"] + scm["E"])[:, 0], scm["B"][:, 0]
            pred11["model"], pred11["model_off"] = (scm["B"] + scm["E"])[:, 1], scm["B"][:, 1]
        ok = P.ok_foot
        day0 = torch.zeros(nb_, dtype=torch.bool, device=dev)
        lam4, lam11 = planck.LAMBDA["viirs_4"], planck.LAMBDA["viirs_11"]
        det = detect(torch.exp(P.logL["viirs"][:, 0]), torch.exp(P.logL["viirs"][:, 1]), common, day0, lam4, lam11)   # fire cells (measured)
        blk = lambda e: P.fp.sum3x3(P.fp.total(e[:, None])[:, 0])
        complete = ok & (P.fp.sum3x3(ok.float()) > 8.5)
        label = blk(det.float() * ok) > 0
        once = once_per_footprint(complete, P.fp)
        meas_pos_cells = det & common
        # eligible MODIS pixels for the hottest-cell rate: complete pixel inside the common mask that holds a VIIRS fire cell
        def has_fire(cells):
            return P.fp._sum(cells.float().reshape(nb_, 1, -1))[:, 0] > 0
        pixok = P.fp.pix_ok.clone(); pixok[:, -1] = False
        elig = {"flagged": pixok & has_fire(det & ok), "fire_level": pixok & has_fire(common & (P.rho4["viirs"] >= FIRE) & ok)}
        Ttrue = torch.exp(tv).reshape(nb_, -1)
        # domain bookkeeping (method independent)
        nvalid = common.flatten(1).sum(1).cpu().numpy(); nobs = meas_pos_cells.flatten(1).sum(1).cpu().numpy()
        for b in range(nb_):
            acc.add(("dom", int(pt[b])), [nvalid[b], nobs[b], 1])
        acc.n_tiles += nb_
        for m in methods:
            L = pred[m].clamp(min=1e-6)
            # ---- blocks (protocol score: predicted block energy)
            sc_blk = blk((L - Bv).clamp(min=0) * ok)
            ii = once.nonzero()
            if len(ii):
                s_, l_ = sc_blk[once].cpu().numpy(), label[once].cpu().numpy(); bi = pt[ii[:, 0].cpu().numpy()]
                for p in np.unique(bi):
                    mk = bi == p; acc.blocks[(m, int(p))][0].append(s_[mk]); acc.blocks[(m, int(p))][1].append(l_[mk])
            # ---- flag rule counts: block [TP, FP, FN, TN, flagged cells, measured cells] (same slots as protocol_eval 'rule') and cell [TP, FP, FN]
            if m != "const":
                dp = detect(L, pred11[m].clamp(min=1e-6), common, day0, lam4, lam11)
                lab_p = blk(dp.float() * ok) > 0
                npred = (dp & common).flatten(1).sum(1).cpu().numpy(); nmeas = nobs
                cnt = [x.flatten(1).sum(1).cpu().numpy() for x in (lab_p & label & once, lab_p & ~label & once, ~lab_p & label & once, ~lab_p & ~label & once)]
                fc = dp & common
                ctp = (fc & meas_pos_cells).flatten(1).sum(1).cpu().numpy(); cfp = (fc & ~meas_pos_cells).flatten(1).sum(1).cpu().numpy(); cfn = (~fc & meas_pos_cells).flatten(1).sum(1).cpu().numpy()
                for b in range(nb_):
                    acc.add(("rule", m, int(pt[b])), [cnt[0][b], cnt[1][b], cnt[2][b], cnt[3][b], npred[b], nmeas[b]])
                    acc.add(("cellrule", m, int(pt[b])), [ctp[b], cfp[b], cfn[b]])
                # ---- fractions skill score
                for k in SCALES:
                    num, den, _ = fss_terms(meas_pos_cells, fc, common, k)
                    num, den = num.cpu().numpy(), den.cpu().numpy()
                    for b in range(nb_):
                        acc.add(("fss", m, k, int(pt[b])), [num[b], den[b]])
            # ---- cell-level precision-recall histogram (score = predicted cell excess over the measured local-median background, as the block score)
            ce = (L - Bv).clamp(min=0)
            for b in range(nb_):
                cm = common[b]; s = ce[b][cm].double(); pos = meas_pos_cells[b][cm]
                bi_ = torch.bucketize(s, edges, right=True)
                hp = torch.bincount(bi_[pos], minlength=NB).cpu().numpy(); hn = torch.bincount(bi_[~pos], minlength=NB).cpu().numpy()
                acc.add(("hist", m, int(pt[b])), np.r_[hp, hn])
            # ---- right hottest cell
            ex, wi, che, chw, tied = hottest_cell(L.reshape(nb_, -1), Ttrue, P.fp.pid, P.fp.n, P.fp.h, P.fp.w)
            for var, el in elig.items():
                ef = el.float()
                agg = torch.stack([(ex * ef).sum(1), (wi * ef).sum(1), (che * ef).sum(1), (chw * ef).sum(1), ef.sum(1), (tied * ef).sum(1)], 1).double().cpu().numpy()
                for b in range(nb_):
                    if agg[b, 4] > 0:
                        acc.add(("hot", m, var, int(pt[b])), agg[b])


# ---------------------------------------------------------------- summary
def _auc(s, l):
    from nightrad.raw.protocol_eval import auc
    return auc(s, l)


def _pair_ids(acc: Acc):
    return sorted({k[-1] for k in acc.d if k[0] == "dom"})


def summarize(acc: Acc, methods=METHODS) -> dict:
    """Everything the page needs, from the accumulators. Aggregation everywhere: pooled over all tiles AND per scene pair then the plain mean over qualifying pairs (spread: per-pair values stored).
    Qualifying pair for blocks / cells / FSS = the protocol's: >= 10 fire blocks and >= 100 other blocks."""
    pairs = _pair_ids(acc)
    methods = [m for m in methods if any(k[0] == "hist" and k[1] == m for k in acc.d)]
    lab = {p: (np.concatenate(acc.blocks[("uniform" if "uniform" in methods else methods[0], p)][1]) if acc.blocks.get(("uniform" if "uniform" in methods else methods[0], p)) else np.array([], bool)) for p in pairs}
    npos = {p: int(lab[p].sum()) for p in pairs}; nneg = {p: int(len(lab[p]) - lab[p].sum()) for p in pairs}
    qual = [p for p in pairs if npos[p] >= MIN_POS_BLOCKS and nneg[p] >= MIN_NEG_BLOCKS]
    out = {"pairs": pairs, "qualifying_pairs": qual, "n_pos_blocks": {str(p): npos[p] for p in pairs}, "n_neg_blocks": {str(p): nneg[p] for p in pairs}, "n_tiles": acc.n_tiles,
           "blocks": {}, "cells": {}, "fss": {}, "hottest": {}, "domain": {}}
    dom = {p: acc.get(("dom", p), 3) for p in pairs}
    out["domain"] = {str(p): {"valid_cells": float(dom[p][0]), "fire_cells": float(dom[p][1]), "tiles": float(dom[p][2]), "fire_cell_fraction": float(dom[p][1] / dom[p][0]) if dom[p][0] else None} for p in pairs}
    tot_valid, tot_fire = sum(dom[p][0] for p in pairs), sum(dom[p][1] for p in pairs)
    out["cell_base_rate"] = float(tot_fire / tot_valid) if tot_valid else None
    for m in methods:
        # ---- blocks
        per, pos_all, neg_all = {}, [], []
        for p in pairs:
            if (m, p) in acc.blocks:
                s = np.concatenate(acc.blocks[(m, p)][0]); l = np.concatenate(acc.blocks[(m, p)][1]).astype(bool)
                per[p] = {"ap": average_precision(s, l), "base_rate": float(l.mean()), "n_pos": int(l.sum()), "n_neg": int((~l).sum())}
                pos_all.append((s, l))
                per[p]["auc"] = _auc(s, l)
        if pos_all:
            S = np.concatenate([a for a, _ in pos_all]); Lb = np.concatenate([b for _, b in pos_all])
            pr, rc, ap, base = pr_curve(S, Lb); cr, cp = thin_curve(pr, rc)
        else:
            ap = base = float("nan"); cr = cp = []
        out["blocks"][m] = {"ap_pooled": ap, "base_rate": base, "n_pos": int(sum(v["n_pos"] for v in per.values())), "n_neg": int(sum(v["n_neg"] for v in per.values())), "curve_recall": cr, "curve_precision": cp,
                            "ap_per_pair": {str(p): v["ap"] for p, v in per.items()}, "auc_per_pair": {str(p): v["auc"] for p, v in per.items()},
                            "auc_pair_mean": float(np.nanmean([per[p]["auc"] for p in qual if p in per])) if any(p in per for p in qual) else None, "ap_pair_mean": float(np.nanmean([per[p]["ap"] for p in qual if p in per])) if any(p in per for p in qual) else None}
        if m != "const":
            r6 = {p: acc.get(("rule", m, p), 6) for p in pairs}; t = sum(r6.values())
            out["blocks"][m]["operating_point"] = {"tp": t[0], "fp": t[1], "fn": t[2], "tn": t[3], "flagged_cells": t[4], "measured_cells": t[5], "precision": float(t[0] / (t[0] + t[1])) if t[0] + t[1] else None,
                                                    "recall": float(t[0] / (t[0] + t[2])) if t[0] + t[2] else None, "csi": csi_from_counts(t[0], t[1], t[2]),
                                                    "flag_count_ratio": float(t[4] / t[5]) if t[5] else None, "csi_per_pair": {str(p): csi_from_counts(*r6[p][:3]) for p in pairs},
                                                    "csi_pair_mean": float(np.nanmean([csi_from_counts(*r6[p][:3]) for p in qual])) if qual else None}
        # ---- cells
        hp, hn = {}, {}
        for p in pairs:
            h = acc.get(("hist", m, p), 2 * NB); hp[p], hn[p] = h[:NB], h[NB:]
        pp = {p: curve_from_counts(hp[p], hn[p]) for p in pairs}
        PP, NN = sum(hp.values()), sum(hn.values())
        pr, rc, ap, base = curve_from_counts(PP, NN); cr, cp = thin_curve(pr, rc)
        out["cells"][m] = {"ap_pooled": ap, "base_rate": base, "n_pos": float(PP.sum()), "n_neg": float(NN.sum()), "curve_recall": cr, "curve_precision": cp, "ap_per_pair": {str(p): pp[p][2] for p in pairs},
                           "ap_pair_mean": float(np.nanmean([pp[p][2] for p in qual])) if qual else None}
        if m != "const":
            c3 = {p: acc.get(("cellrule", m, p), 3) for p in pairs}; t = sum(c3.values())
            out["cells"][m]["operating_point"] = {"tp": t[0], "fp": t[1], "fn": t[2], "precision": float(t[0] / (t[0] + t[1])) if t[0] + t[1] else None, "recall": float(t[0] / (t[0] + t[2])) if t[0] + t[2] else None,
                                                   "csi": csi_from_counts(*t), "csi_per_pair": {str(p): csi_from_counts(*c3[p]) for p in pairs}, "csi_pair_mean": float(np.nanmean([csi_from_counts(*c3[p]) for p in qual])) if qual else None}
            # ---- fss
            out["fss"][m] = {}
            for k in SCALES:
                nd = {p: acc.get(("fss", m, k, p), 2) for p in pairs}
                per = {p: fss_value(*nd[p]) for p in pairs}
                tn, td = sum(v[0] for v in nd.values()), sum(v[1] for v in nd.values())
                out["fss"][m][str(k)] = {"pooled": fss_value(tn, td), "pair_mean": float(np.nanmean([per[p] for p in qual])) if qual else None, "per_pair": {str(p): per[p] for p in pairs},
                                         "pair_sd": float(np.nanstd([per[p] for p in qual])) if qual else None}
        # ---- hottest cell
        out["hottest"][m] = {}
        for var in HOT_VARIANTS:
            h = {p: acc.get(("hot", m, var, p), 6) for p in pairs}; t = sum(h.values())
            if t[4] == 0:
                out["hottest"][m][var] = {"n_pixels": 0}; continue
            ok_p = [p for p in pairs if h[p][4] >= MIN_PIXELS]
            out["hottest"][m][var] = {"n_pixels": float(t[4]), "n_pairs": len(ok_p), "exact_pooled": float(t[0] / t[4]), "within1_pooled": float(t[1] / t[4]), "chance_exact": float(t[2] / t[4]), "chance_within1": float(t[3] / t[4]),
                                      "truth_tied_share": float(t[5] / t[4]), "exact_pair_mean": float(np.mean([h[p][0] / h[p][4] for p in ok_p])) if ok_p else None,
                                      "within1_pair_mean": float(np.mean([h[p][1] / h[p][4] for p in ok_p])) if ok_p else None,
                                      "exact_per_pair": {str(p): float(h[p][0] / h[p][4]) for p in pairs if h[p][4] > 0}, "within1_per_pair": {str(p): float(h[p][1] / h[p][4]) for p in pairs if h[p][4] > 0},
                                      "chance_exact_per_pair": {str(p): float(h[p][2] / h[p][4]) for p in pairs if h[p][4] > 0}, "chance_within1_per_pair": {str(p): float(h[p][3] / h[p][4]) for p in pairs if h[p][4] > 0},
                                      "n_pixels_per_pair": {str(p): float(h[p][4]) for p in pairs if h[p][4] > 0}}
    return out


def file_sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
