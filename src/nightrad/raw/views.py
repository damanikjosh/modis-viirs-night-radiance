"""Fixed sensor views of a fine-cell scene (no learned weights).

Scene: per fine cell, background radiance B (4, 11 um) and non-negative excess E, both in the VIIRS band (I4 3.74, I5 11.45 um).
 VIIRS view : the cell itself, L = B + E.
 MODIS view : mean of B and of E over the native pixel's cells (scatter-mean via source-pixel id, vectorised over the batch),
              then band conversion: background through Planck (same temperature), excess by the fixed factor at T_fire.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nightrad.raw import planck
from nightrad.raw.mask import Footprints


@dataclass
class ViewConfig:
    t_fire: float = planck.T_FIRE_DEFAULT
    kernel: str = "tri"  # DEFAULT: weighted footprint reaching into neighbouring native pixels; 'box' = old plan variant (own pixel only)
    w_scan: float = 0.001  # FITTED on ALL real v3 TRAIN tiles that pass the 10 % mask (3,906; scripts/physics_real.py, fire-bearing footprints): w_scan 0.0011, w_track -0.0435 clipped to 0 = effectively the box view. A fit on the first 2,000 train tiles gave 0.048 / -0.003; the physics-review triangle (0.125) is not supported. Frozen in configs/raw.json; never learned by the model
    w_track: float = 0.0
    tie11: bool = False


class ModisView:
    """Holds the band constants; learned_corr (optional fallback, off by default) = per-band log gain on the background."""

    def __init__(self, cfg: ViewConfig | None = None):
        self.cfg = cfg or ViewConfig()
        L = planck.LAMBDA
        self.l4 = (L["viirs_4"], L["modis_4"])
        self.l11 = (L["viirs_11"], L["modis_11"])
        self.f4 = planck.excess_factor(*self.l4, self.cfg.t_fire)
        self.f11 = planck.excess_factor(*self.l11, self.cfg.t_fire)

    def __call__(self, B: torch.Tensor, E: torch.Tensor, fp: Footprints, log_gain: torch.Tensor | None = None):
        """B, E [N,2,H,W] fp32 linear radiance (VIIRS band) -> MODIS-band radiance [N,2,H,W], constant within a footprint."""
        if self.cfg.kernel == "tri" and fp.nbr is not None:
            mB, mE = fp.tri_mean(B, self.cfg.w_scan, self.cfg.w_track), fp.tri_mean(E, self.cfg.w_scan, self.cfg.w_track)
        else:  # box (or tri requested but no neighbour table in this batch)
            mB, mE = fp.mean(B), fp.mean(E)
        nb = B.shape[1]  # 2 = both bands; 1 = 4 um only
        bg = torch.stack([planck.convert_background(mB[:, 0], *self.l4)] + ([planck.convert_background(mB[:, 1], *self.l11)] if nb == 2 else []), 1)
        if log_gain is not None:
            bg = bg * torch.exp(log_gain)[None, :, None, None]
        f = torch.tensor([self.f4, self.f11][:nb], device=B.device, dtype=B.dtype)[None, :, None, None]
        return bg + f * mE

    def excess(self, E: torch.Tensor, fp: Footprints) -> torch.Tensor:
        """Excess part of the MODIS view only (for the energy-conservation checks)."""
        f = torch.tensor([self.f4, self.f11], device=E.device, dtype=E.dtype)[None, :, None, None]
        return f * fp.mean(E)


def viirs_view(B: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    return B + E


def modis_to_viirs_background(L: torch.Tensor, nb: int = 2) -> torch.Tensor:
    """MODIS-band background radiance [N,2,H,W] -> VIIRS band (Planck, same temperature). nb=1: only the 4 um band is converted and returned [N,1,H,W]."""
    l = planck.LAMBDA
    out = [planck.convert_background(L[:, 0], l["modis_4"], l["viirs_4"])]
    if nb == 2:
        out.append(planck.convert_background(L[:, 1], l["modis_11"], l["viirs_11"]))
    return torch.stack(out, 1)


def tie_kappa(t_tie: float = 800.0) -> float:
    """E11/E4 in the VIIRS band for a blackbody fire at t_tie (background neglected): B(11.45, T) / B(3.74, T)."""
    L = planck.LAMBDA
    f = lambda lam: planck.C1 / (lam**5 * math.expm1(planck.C2 / (lam * t_tie)))
    return f(L["viirs_11"]) / f(L["viirs_4"])
