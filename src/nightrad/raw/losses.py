"""Four reconstruction terms (M->M, V->V with hiding; V->M, M->V) on log radiance, inside the common mask.

Each ingredient is a small configurable piece: huber(), fire_weight(), term_loss(), alloc_loss().
Residual r = ln L_pred - ln L_meas. Because both are compared against the same background, r equals the log-ratio-to-background
residual of the excess (the plan's 'excess as log-ratio'), so no separate excess term is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from nightrad.raw.mask import Footprints


@dataclass
class LossConfig:
    delta: float = 0.10  # Huber delta on ln L (~2.3 K at 4 um, 7 K at 11 um)
    fire_gain: float = 4.0  # w = 1 + gain * clip(rho4, 0, fire_clip), rho4 measured on the target
    fire_clip: float = 1.5
    hidden_weight: float = 1.0  # same-sensor terms: weight of hidden cells ...
    visible_weight: float = 0.25  # ... and of visible (trivially copyable) cells
    # purpose in the weights: M->V is the learned product, V->M the physics check, same-sensor terms are low-weight regularisers
    term_weights: dict = field(default_factory=lambda: {"mv": 1.0, "vm": 0.5, "mm": 0.1, "vv": 0.1})
    footprint_norm: str = "area"  # MODIS-target terms: 'area' (a footprint counts as its n cells) or 'unit' (counts 1)
    band_weights: tuple = (1.0, 1.0)  # 4 um, 11 um
    fire_level: float = 0.5  # rho4 >= this = 'fire-level' cell for logging
    alloc_weight: float = 0.1
    class_balance: bool = True  # DEFAULT: per-group mean error (bg / near-fire / fire-level, from the RAW radiance of the target sensor) x fixed shares; False = old fire weighting
    class_shares: tuple = (0.4, 0.2, 0.4)
    near_radius: int = 2
    hurdle_weight: float = 1.0
    block_w_mv: float = 0.0  # SWEEP-1 switch (0 = off = reference): weight of the 3x3-block mass term for M->V (core.block_loss)
    block_w_vm: float = 0.0  # same for V->M
    w_energy: float = 0.0  # SWEEP-1 (review C1): ENERGY-CONSISTENCY term, 0 = off: scene decoded from MODIS (clean), pushed through the MODIS view, must reproduce the measured MODIS radiance per footprint (same group shares as the other terms)
    mode: str = "legacy"  # 'legacy' = group shares + Huber on log (reference) | 'gamma' = mean-consistent Gamma deviance on LINEAR radiance, NO target-dependent weights, scales cell/pixel/3x3 block
    input_weight: float = 0.0  # gamma mode: extra weight on M->V cells under a bright MODIS pixel (weight depends on the INPUT side only, never on the target); 0 = off
    loss_family: str = "gamma"  # gamma | poisson | nmse (mode "gamma" only; everything else identical across families)
    bright_weight: float = 0.0  # (gamma mode): smooth INPUT-side weight w = 1 + G * clamp(rho4_modis, 0, 3) on every M->V term (cell / pixel / block), rho4 = ln(MODIS radiance / its local median); no target-dependent weight; 0 = off
    scale_w: tuple = (1.0, 1.0, 1.0)  # gamma mode, M->V path: weights of the fine-cell / MODIS-pixel / 3x3-block deviance
    false_share: float = 0.0  # SWEEP-1 (review C1): share of a separate 4th loss group in the M->V term = cells under a bright MODIS pixel where VIIRS shows no fire-level excess (false excess); the other three shares are scaled by 1 - false_share
    block_w_mv: float = 0.0  # NEW (sweep 1, not approved): weight of the 3x3-block loss on sums of excess, M->V and V->M; 0 = off (reference)
    block_w_vm: float = 0.0


TAP = None  # diagnostics only (nightrad.raw.diag): when a list, every loss term appends its per-cell contribution [B,H,W] (sums to the term's loss); None = no change to training


def shares_for_fire(f: float) -> tuple:
    """Pre-declared grid fire share 0.2 / 0.4 / 0.6; the remainder split background : near-fire = 2 : 1 -> (0.4, 0.2, 0.4) at f = 0.4."""
    r = 1.0 - f
    return (r * 2 / 3, r / 3, f)


def huber(r: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber / delta so the gradient magnitude is ~1 outside the quadratic zone."""
    a = r.abs()
    return torch.where(a < delta, 0.5 * r * r / delta, a - 0.5 * delta)


def fire_weight(rho4: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    return 1.0 + cfg.fire_gain * rho4.clamp(0.0, cfg.fire_clip)


def classes(rho4, mask, cfg: LossConfig):
    """(background, near-fire, fire-level) bool masks inside `mask`. Fire-level: measured rho4 >= cfg.fire_level; near: within near_radius."""
    fire = mask & (rho4 >= cfg.fire_level)
    r = cfg.near_radius
    near = (F.max_pool2d(fire.float()[:, None], 2 * r + 1, 1, r)[:, 0] > 0) & mask & ~fire
    return mask & ~fire & ~near, near, fire


def term_loss(pred_log, tgt_log, mask, w_base, sat4, cfg: LossConfig, rho4, false_mask=None):
    """pred_log, tgt_log [B,2,H,W] fp32; mask [B,H,W] bool; w_base [B,H,W] (hidden/visible, area weights); sat4 [B,H,W] bool: 4 um
    target saturated -> one-sided (only pred < target is penalised). Weighting: fire weight 1+gain*rho4 (plan default) OR, with
    cfg.class_balance, a fixed share per class. Returns (scalar, detached stats incl. per-class loss shares of the objective)."""
    r = pred_log - torch.where(mask[:, None], tgt_log, pred_log.detach())  # no gradient/value from masked cells
    r4 = torch.where(sat4, r[:, 0].clamp(max=0.0), r[:, 0])
    nb = r.shape[1]  # 2 bands, or 1 (4 um only, ): then tgt_log carries the 4 um band alone
    r = torch.stack([r4, r[:, 1]], 1) if nb == 2 else r4[:, None]
    h = huber(r, cfg.delta)
    bw = torch.tensor(cfg.band_weights[:nb], device=h.device, dtype=h.dtype)[None, :, None, None]
    cell = torch.where(mask, (h * bw).sum(1) / bw.sum(), torch.zeros_like(h[:, 0]))  # masked cells are exactly 0, never NaN*0
    mf = mask.float()
    cls = classes(rho4, mask, cfg)
    names = ("bg", "near", "fire")
    shares = tuple(cfg.class_shares)
    if cfg.false_share > 0 and false_mask is not None:  # 4th group: cells of a bright MODIS pixel without VIIRS fire-level excess
        fm = false_mask & mask & ~cls[2]
        cls = (cls[0] & ~fm, cls[1] & ~fm, cls[2], fm)
        names = ("bg", "near", "fire", "false")
        shares = tuple(x * (1 - cfg.false_share) for x in shares) + (cfg.false_share,)
    if cfg.class_balance:
        cnt = [c.float().sum() for c in cls]
        present = torch.stack([(c > 0).float() for c in cnt])
        sh = torch.tensor(shares, device=h.device) * present
        sh = sh / sh.sum().clamp(min=1e-9)
        per = [(cell * w_base * m_.float()).sum() / c_.clamp(min=1.0) for c_, m_ in zip(cnt, cls)]
        loss = sum(sh[i] * per[i] for i in range(len(cls)))
        if TAP is not None:
            TAP.append(sum(sh[i] * cell * w_base * cls[i].float() / cnt[i].clamp(min=1.0) for i in range(len(cls))))
        eff = [sh[i] * per[i] for i in range(len(cls))]
    else:
        w = w_base * fire_weight(rho4, cfg)
        n = mf.sum().clamp(min=1.0)
        loss = (cell * w * mf).sum() / n
        if TAP is not None:
            TAP.append(cell * w * mf / n)
        eff = [(cell * w * c.float()).sum() / n for c in cls]
    with torch.no_grad():
        tot = sum(e.detach() for e in eff).clamp(min=1e-12)
        raw = [(cell * c.float()).sum() for c in cls]
        rt = sum(raw).clamp(min=1e-12)
        fl = (cell * cls[2]).sum() / cls[2].sum().clamp(min=1)
        bl = (cell * cls[0]).sum() / cls[0].sum().clamp(min=1)
        stats = {"loss": loss.detach(), "fire_cell": fl, "bg_cell": bl, "n": mf.sum(), "n_fire": cls[2].sum().float(),
                 "abs_dlog4_fire": _med(r[:, 0].abs(), cls[2]), "abs_dlog11": _med(r[:, 1].abs(), mask) if nb == 2 else h.new_zeros(())}
        for i, nme in enumerate(names):
            stats[f"share_{nme}"] = eff[i].detach() / tot  # share of the optimised objective
            stats[f"rawshare_{nme}"] = raw[i] / rt  # share of the plain unweighted loss
    return loss, stats


def hurdle_loss(h_cell, a_cell, e_meas, fp: Footprints, ok, pos_cell, cfg: LossConfig):
    """Footprint total as (probability of fire) x (amount). h_cell, a_cell [B,H,W] raw logits (footprint-mean taken here);
    e_meas [B,H,W] measured VIIRS 4 um excess per cell (>=0); pos_cell [B,H,W] bool measured fire-level VIIRS cell.
    NLL = BCE(sigmoid(h), 1[footprint has a fire-level cell]) + 1[pos] * Huber(a - ln mean excess per cell). Area-weighted over ok cells."""
    h = fp.mean(h_cell[:, None])[:, 0]
    a = fp.mean(a_cell[:, None])[:, 0]
    pos = fp.any(pos_cell & ok)
    m = fp.mean(torch.where(ok, e_meas, torch.zeros_like(e_meas))[:, None])[:, 0]
    okf = ok.float()
    bce = F.binary_cross_entropy_with_logits(h, pos.float(), reduction="none")
    amt = huber(a - torch.log(m.clamp(min=1e-4)), 0.5)
    n = okf.sum().clamp(min=1.0)
    loss = ((bce + pos.float() * amt) * okf).sum() / n
    return loss, {"hurdle/bce": (bce * okf).sum().detach() / n, "hurdle/amount": (pos.float() * amt * okf).sum().detach() / n,
                  "hurdle/pos_frac": (pos.float() * okf).sum().detach() / n}


def _med(x, m):
    v = x[m]
    return v.median() if v.numel() else x.new_zeros(())


def alloc_loss(logit, e_meas, fp: Footprints, ok, min_mass: float = 0.0):
    """Stage-3 allocation likelihood: softmax over each footprint's cells; soft-label cross-entropy to the measured VIIRS excess share
    q_c = E_c / sum_f E, weighted by the footprint's excess mass (normalised). Returns (loss nats per unit mass, uniform-baseline nats)."""
    e = torch.where(ok, e_meas.clamp(min=0.0), torch.zeros_like(e_meas))
    mass_f = fp.total(e[:, None])[:, 0]  # per cell: footprint mass
    cnt = fp.total(torch.ones_like(e)[:, None])[:, 0]
    logp = logit - fp.logsumexp(logit)
    # sum over cells of E_c * (-logp_c) = mass * CE ; each footprint's total weight = its mass
    num = (e * (-logp) * ok).sum()
    uni = (e * torch.log(cnt.clamp(min=1.0)) * ok).sum()
    den = (e * ok).sum().clamp(min=1e-12)
    return num / den, (uni / den).detach()


def apply_allocation(E: torch.Tensor, logit: torch.Tensor, fp: Footprints) -> torch.Tensor:
    """Redistribute each footprint's excess mass over its cells with softmax(logit): conserves the footprint sum exactly. E [N,2,H,W]."""
    p = torch.exp(logit - fp.logsumexp(logit))[:, None]
    return fp.total(E) * p


def gamma_dev(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gamma deviance d(y, mu) = y/mu - ln(y/mu) - 1 >= 0 (0 iff y = mu); mean-consistent (Bregman) for the conditional mean; fp32; mu, y clamped positive."""
    mu = mu.float().clamp(min=1e-6); y = y.float().clamp(min=1e-6)
    r = y / mu
    return r - torch.log(r) - 1.0


def family_dev(mu, y, cfg: LossConfig, ref=None, bg=None):
    """Per-element deviance of the chosen family, all >= 0 and 0 at mu = y, all minimised at the conditional MEAN, no target-dependent weight:
      gamma  : y/mu - ln(y/mu) - 1
      poisson: y ln(y/mu) - (y - mu), y and mu divided by a FIXED reference (Planck 300 K radiance of that band: a constant, not the target)
      nmse   : ((y - mu)/b)^2 with b = the local masked-MEDIAN background radiance of the target sensor's own measured field (already used for the input features; detached; a median
               over the neighbourhood, so it carries the fire content of the target cell only as much as a median does)"""
    f = cfg.loss_family
    if f == "gamma":
        return gamma_dev(mu, y)
    mu = mu.float().clamp(min=1e-6); y = y.float().clamp(min=1e-6)
    if f == "poisson":
        r = ref.to(mu.device).float()[None, :, None, None].clamp(min=1e-6)
        mu, y = mu / r, y / r
        return (y * torch.log(y / mu) - (y - mu)).clamp(min=0)
    if f == "nmse":
        return ((y - mu) / bg.detach().float().clamp(min=1e-3)) ** 2
    raise ValueError(f)


def gamma_term(mu, y, mask, w, sat4, cfg: LossConfig, ref=None, bg=None):
    """mu, y [B,2,H,W] linear radiance (4, 11 um); mask [B,H,W] bool; w [B,H,W] weights that depend on the INPUT only (hiding / footprint area), never on the target;
    sat4 [B,H,W] bool: 4 um target saturated -> one-sided (no penalty when mu >= y). Mean over the mask of the band-averaged deviance."""
    d = family_dev(mu, y, cfg, ref, bg)
    d4 = torch.where(sat4 & (mu[:, 0].float() >= y[:, 0].float()), torch.zeros_like(d[:, 0]), d[:, 0])
    nb = d.shape[1]  # 2, or 1 (4 um only, )
    bw = torch.tensor(cfg.band_weights[:nb], device=d.device, dtype=d.dtype)
    cell = (d4 * bw[0] + d[:, 1] * bw[1]) / bw.sum() if nb == 2 else d4 * bw[0] / bw.sum()
    ww = torch.where(mask, w, torch.zeros_like(w))
    if TAP is not None:
        TAP.append(torch.where(mask, cell, torch.zeros_like(cell)) * ww / ww.sum().clamp(min=1e-9))
    return (torch.where(mask, cell, torch.zeros_like(cell)) * ww).sum() / ww.sum().clamp(min=1e-9)
