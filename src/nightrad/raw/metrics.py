"""Physical baselines (no learned weights) and fire-excess metrics, usable on real data (measured fire-level) and synthetic (truth)."""
from __future__ import annotations

import numpy as np
import torch

from nightrad.raw.core import Prep, RunConfig
from nightrad.raw.views import ModisView, modis_to_viirs_background


def baseline_scenes(P: Prep, cfg: RunConfig):
    """Learning-free scenes. V-sourced: B = local median, E = (L - B)^+ . M-sourced: same in the MODIS band, converted to the VIIRS
    band (background by Planck, excess by 1/factor); excess spread UNIFORMLY over the footprint (what the data say without a model)."""
    mv = ModisView(cfg.view)
    out = {}
    Lv = torch.exp(P.logL["viirs"]); Bv = torch.exp(P.med["viirs"])
    out["viirs"] = (Bv, (Lv - Bv).clamp(min=0))
    Lm = torch.exp(P.logL["modis"]); Bm = torch.exp(P.med["modis"])
    f = torch.tensor([mv.f4, mv.f11], device=Lm.device)[None, :, None, None]
    out["modis"] = (modis_to_viirs_background(Bm), (Lm - Bm).clamp(min=0) / f)  # footprint-constant already
    return out


def bg_only_pred_modis(P: Prep, cfg: RunConfig):
    """'Background-only predictor': MODIS view of the V-sourced scene with the excess set to 0."""
    mv = ModisView(cfg.view)
    Bv, _ = baseline_scenes(P, cfg)["viirs"]
    return mv(Bv, torch.zeros_like(Bv), P.fp)


def physical_pred_modis(P: Prep, cfg: RunConfig):
    mv = ModisView(cfg.view)
    B, E = baseline_scenes(P, cfg)["viirs"]
    return mv(B, E, P.fp)


class Acc:
    """Collects per-cell absolute errors by group; median at the end (CPU)."""

    def __init__(self):
        self.d: dict[str, list] = {}

    def add(self, name, vals: torch.Tensor):
        self.d.setdefault(name, []).append(vals.detach().float().cpu())

    def median(self, name):
        v = torch.cat(self.d[name]) if self.d.get(name) else torch.zeros(0)
        return float(v.median()) if v.numel() else float("nan")

    def mean(self, name):
        v = torch.cat(self.d[name]) if self.d.get(name) else torch.zeros(0)
        return float(v.mean()) if v.numel() else float("nan")

    def n(self, name):
        return int(sum(x.numel() for x in self.d.get(name, [])))

    def cat(self, name):
        return torch.cat(self.d[name]) if self.d.get(name) else torch.zeros(0)


def _split_add(acc: Acc, name: str, vals: torch.Tensor, sel: torch.Tensor, day: torch.Tensor):
    """Day/night is a first-class split: every named group is also stored as name_day / name_night. vals/sel [B,H,W], day [B,1,1] bool."""
    acc.add(name, vals[sel])
    acc.add(name + "_day", vals[sel & day])
    acc.add(name + "_night", vals[sel & ~day])


@torch.no_grad()
def evaluate_batch(aux, inp, cfg: RunConfig, acc: Acc, ev: dict | None = None):
    """Accumulate fire-excess errors from forward_loss(return_aux=True). Fire-level = MEASURED rho4 >= fire_level on the target.
    Groups (each also _day/_night): vm_fire/vm_bg (V->M footprints), mv_fire/mv_bg (M->V cells), vv_hidden_*, *_base = calibrated-free
    physical baseline, vm_fire_bgonly = background-only predictor, mass_* = predicted vs measured excess mass (V-clean scene, fire-level
    cells; the 'excess collapse' guard), scene_gap = |ln(sum E_M / sum E_V)| over 3x3 native pixels (diagnostic, never trained),
    ratio3x3 = predicted / measured MODIS excess on 3x3 pixel sums. If ev has eval_E4 (synthetic truth): truth recovery."""
    P, preds, hidden = aux["P"], aux["preds"], aux["hidden"]
    fl = cfg.loss.fire_level
    day = P.day[:, None, None]
    add = lambda name, vals, sel: _split_add(acc, name, vals, sel, day)
    sc_v, sc_m = aux["scenes"][("viirs", "clean")], aux["scenes"][("modis", "clean")]
    base_vm = torch.log(physical_pred_modis(P, cfg).clamp(min=1e-12))
    bg_vm = torch.log(bg_only_pred_modis(P, cfg).clamp(min=1e-12))
    bs = baseline_scenes(P, cfg)
    Bm, Em = bs["modis"]
    base_mv = torch.log((Bm + P.fp.mean(Em)).clamp(min=1e-12))  # uniform spread of the footprint excess over its cells
    tm, tv = P.logL["modis"][:, 0], P.logL["viirs"][:, 0]
    fire_m = P.ok_foot & (P.rho4["modis"] >= fl) & (inp["modis_sat"] == 0)
    bgm = P.ok_foot & (P.rho4["modis"] < fl)
    fire_v = P.common & (P.rho4["viirs"] >= fl) & (inp["viirs_sat"] == 0)
    bgv = P.common & (P.rho4["viirs"] < fl)
    ae = lambda p, t: (p[:, 0] - t).abs()
    add("vm_fire", ae(preds["vm"], tm), fire_m); add("vm_bg", ae(preds["vm"], tm), bgm)
    add("vm_fire_base", ae(base_vm, tm), fire_m); add("vm_fire_bgonly", ae(bg_vm, tm), fire_m); add("vm_bg_base", ae(base_vm, tm), bgm)
    add("mv_fire", ae(preds["mv"], tv), fire_v); add("mv_bg", ae(preds["mv"], tv), bgv); add("mv_fire_base", ae(base_mv, tv), fire_v)
    add("mv_bg_base", ae(base_mv, tv), bgv)
    hv = hidden["viirs"] & fire_v
    add("vv_hidden_fire", ae(preds["vv"], tv), hv)
    hb = hidden["viirs"] & bgv
    add("vv_hidden_bg", ae(preds["vv"], tv), hb)
    r11 = (preds["vv"][:, 1] - P.logL["viirs"][:, 1]).abs()
    tile_med = torch.stack([P.logL["viirs"][b, 1][P.common[b]].median() if P.common[b].any() else torch.zeros((), device=r11.device)
                            for b in range(r11.shape[0])])[:, None, None]
    add("vv_hidden_bg11", r11, hb); add("vv_hidden_bg11_tilemedian", (tile_med - P.logL["viirs"][:, 1]).abs(), hb)
    # excess-mass guard (measured only): predicted V-clean excess mass vs measured excess mass on fire-level cells
    Bv = torch.exp(P.med["viirs"][:, 0])
    meas_e = (torch.exp(tv) - Bv).clamp(min=0)
    add("mass_pred", sc_v["E"][:, 0], fire_v); add("mass_meas", meas_e, fire_v)
    # scene gap between the scene decoded from MODIS and from VIIRS, at 3x3 native-pixel sums (measured, not trained)
    if P.fp.nbr is not None:
        sel = P.ok_foot & fire_m
        sM = P.fp.sum3x3(sc_m["E"][:, 0]); sV = P.fp.sum3x3(sc_v["E"][:, 0])
        add("scene_gap", torch.log((sM + 1e-4) / (sV + 1e-4)).abs(), sel)
        mv = ModisView(cfg.view)
        Lp = mv(sc_v["B"], sc_v["E"], P.fp)[:, 0]
        Bmed = torch.exp(P.med["modis"][:, 0])
        pe3, me3 = P.fp.sum3x3((Lp - Bmed).clamp(min=0)), P.fp.sum3x3((torch.exp(tm) - Bmed).clamp(min=0))
        add("ratio3x3_pred", pe3, sel); add("ratio3x3_meas", me3, sel)
    if P.fp.nbr is not None:
        # ~3 km scale (3x3 native pixels), the scale at which M->V is claimed: |ln(sum predicted excess / sum measured excess)| per block
        w = P.ok_foot
        eps = 1e-3
        pe_m = (torch.exp(preds["vm"][:, 0]) - Bmed).clamp(min=0); me_m = (torch.exp(tm) - Bmed).clamp(min=0)
        add("vm3_fire", torch.log((P.fp.sum3x3(pe_m) + eps) / (P.fp.sum3x3(me_m) + eps)).abs(), w & fire_m)
        pe_phys = (torch.exp(base_vm[:, 0]) - Bmed).clamp(min=0)
        add("vm3_fire_base", torch.log((P.fp.sum3x3(pe_phys) + eps) / (P.fp.sum3x3(me_m) + eps)).abs(), w & fire_m)
        pe_v = (torch.exp(preds["mv"][:, 0]) - Bv).clamp(min=0) * P.ok_foot; me_v = meas_e * P.ok_foot
        blk = lambda e: P.fp.sum3x3(P.fp.total(e[:, None])[:, 0])
        fire_blk = w & P.fp.any(fire_v & P.ok_foot)
        add("mv3_fire", torch.log((blk(pe_v) + eps) / (blk(me_v) + eps)).abs(), fire_blk)
        mvb = lambda Bv_, Es: blk(Es)
        # trivial baseline at 3 km: the MODIS excess itself converted to the VIIRS band (spreading inside a pixel does not change block sums)
        add("mv3_fire_base", torch.log((blk((Em).clamp(min=0)[:, 0] * P.ok_foot) + eps) / (blk(me_v) + eps)).abs(), fire_blk)
    if ev is not None and "eval_E4" in ev:
        E4t = ev["eval_E4"].float()
        sel = P.common & (inp["viirs_sat"] == 0)
        big = sel & (E4t > 0.3 * Bv)
        acc.add("rec_ratio_pred", sc_v["E"][:, 0][big]); acc.add("rec_ratio_true", E4t[big])


_KEYS = ("vm_fire", "vm_fire_base", "vm_fire_bgonly", "vm_bg", "vm_bg_base", "mv_fire", "mv_fire_base", "mv_bg",
         "vv_hidden_fire", "vv_hidden_bg", "vv_hidden_bg11", "vv_hidden_bg11_tilemedian", "scene_gap", "vm3_fire", "mv3_fire", "mv3_fire_base", "mv_bg_base", "vm3_fire_base")


def summarize(acc: Acc) -> dict:
    s = {}
    for k in _KEYS:
        for suf in ("", "_day", "_night"):
            if acc.d.get(k + suf):
                s[k + suf] = acc.median(k + suf)
    for k in _KEYS:
        s.setdefault(k, float("nan"))
    s["n_vm_fire"], s["n_mv_fire"] = acc.n("vm_fire"), acc.n("mv_fire")
    s["n_vm_fire_day"], s["n_vm_fire_night"] = acc.n("vm_fire_day"), acc.n("vm_fire_night")
    # headline for 'best by fire excess': the 3x3-block fire-level excess errors (M->V claimed at ~3 km, V->M physics check) when the
    # neighbour table exists, else the per-pixel fire-level medians (lower better)
    parts = [s["vm3_fire"], s["mv3_fire"]]
    parts = [p for p in parts if p == p]
    if not parts:
        parts = [p for p in (s["vm_fire"], s["mv_fire"], s["vv_hidden_fire"]) if p == p]
    s["fire_err"] = float(np.mean(parts)) if parts else float("inf")
    for suf in ("", "_day", "_night"):
        if acc.n("mass_meas" + suf):
            s["excess_mass_ratio_meas" + suf] = float(acc.cat("mass_pred" + suf).sum() / acc.cat("mass_meas" + suf).sum().clamp(min=1e-12))
    if acc.n("ratio3x3_meas"):
        s["ratio3x3"] = float(acc.cat("ratio3x3_pred").sum() / acc.cat("ratio3x3_meas").sum().clamp(min=1e-12))
    if acc.n("rec_ratio_true"):
        p, t = acc.cat("rec_ratio_pred"), acc.cat("rec_ratio_true")
        s["excess_mass_ratio"] = float(p.sum() / t.sum().clamp(min=1e-12))
        s["rank_corr"] = _spearman(p, t)
    return s


def excess_collapsed(s: dict, floor: float = 0.2) -> bool:
    """Stop rule: predicted excess mass on measured fire-level cells fell below `floor` x the measured mass (excess ~ 0)."""
    r = s.get("excess_mass_ratio_meas")
    return r is not None and r == r and r < floor


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 3:
        return float("nan")
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp(min=1e-12))


def check_degenerate(s: dict, margin: float = 0.30, mass_range=(0.5, 2.0), min_rank=0.4) -> list[str]:
    """Return the list of FAILED degenerate-solution checks (empty = passes). Designed to FAIL for excess == 0 and for input copying.
      1. beats the background-only predictor on V->M fire footprints by >= margin  (fails if excess is ~0)
      2. recovered excess mass within mass_range of truth and rank corr >= min_rank (fails if excess is ~0 / flat)
      3. hidden 11 um background beats the tile-median predictor by >= margin   (fails if it ignores context / copies zeros)
      4. M->V fire cells within 25 % of the uniform-spread baseline (allocation is learned slowly; reported, loosely checked)."""
    f = []
    if not s["vm_fire"] <= (1 - margin) * s["vm_fire_bgonly"]:
        f.append(f"vm_fire {s['vm_fire']:.3f} not <= {(1 - margin):.2f} x background-only {s['vm_fire_bgonly']:.3f}")
    if "excess_mass_ratio" in s:
        if not mass_range[0] <= s["excess_mass_ratio"] <= mass_range[1]:
            f.append(f"excess mass ratio {s['excess_mass_ratio']:.3f} outside {mass_range}")
        if not s["rank_corr"] >= min_rank:
            f.append(f"rank corr {s['rank_corr']:.3f} < {min_rank}")
    if not s["vv_hidden_bg11"] <= (1 - margin) * s["vv_hidden_bg11_tilemedian"]:
        f.append(f"hidden 11um error {s['vv_hidden_bg11']:.4f} not <= {(1 - margin):.2f} x tile-median {s['vv_hidden_bg11_tilemedian']:.4f}")
    if not s["mv_fire"] <= 1.25 * s["mv_fire_base"]:
        f.append(f"mv_fire {s['mv_fire']:.3f} more than 25 % worse than the uniform-spread baseline {s['mv_fire_base']:.3f}")
    return f
