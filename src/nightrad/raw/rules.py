"""ONE documented detection rule, applied with the SAME code to predicted radiance and to measured radiance. A metric only, never a training target.

Published rule (what the code follows; Giglio et al. 2016 for MODIS, Schroeder et al. 2014 for VIIRS):
  MODIS MOD14 C6 (Giglio et al. 2016): potential-fire test T4 > T4*, dT > dT* with T4* = mean(T4_bg)+5 K clamped to [300, 330] K, dT* = mean(dT_bg)+5 K clamped
  to [10, 35] K (C5: fixed 310/305 K and 10 K); absolute test T4 >= 360 K day / 320 K night (Eq. 1); contextual tests dT > dT_bg + max(3.5 sigma, 6 K) and
  T4 > T4_bg + 3 sigma, T11 > T11_bg + sigma11 - 4 K (Eq. 2-6, structure verified, constants partly claimed).
  VIIRS 375 m (Schroeder et al. 2014): night absolute T4 >= 320 K; candidate potential level night 295 K (day 325 K), context otherwise as MODIS (the VIIRS daytime
  algorithm is NOT reproduced: not a fixed threshold in the paper).
IMPLEMENTED here: potential-fire gate (C6 dynamic or C5 fixed), absolute test, contextual dT / T4 / T11 tests with background mean and sigma over valid non-potential cells of a
  dilated k x k window on the cell grid (min_bg valid cells), common-mask validity only.
NOT implemented: window growth (11x11 -> 31x31 / 21x21), background exclusion of cloud-edge/water/sun-glint/saturated pixels, the 5-K-type extra test, daytime rejection
  (sun glint, desert boundary, coast, Giglio 2016 sec 3.6), all quality flags, the 2-pixel adjacency rules, VIIRS day potential/contextual details, FRP.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nightrad.raw import planck


@dataclass
class RuleConfig:
    mode: str = "c6"  # 'c6' dynamic potential-fire thresholds, 'c5' fixed ones
    t4_pot: tuple = (305.0, 310.0)  # c5 (night, day)
    dt_pot: tuple = (10.0, 10.0)
    t4_abs: tuple = (320.0, 360.0)  # absolute test (night, day) MODIS
    t4_abs_viirs: tuple = (320.0, 1e9)  # VIIRS: night 320 K; day has no fixed test
    t4_pot_viirs: tuple = (295.0, 325.0)
    t4_star: tuple = (300.0, 330.0)  # c6 clamp of mean(T4)+5
    dt_star: tuple = (10.0, 35.0)  # c6 clamp of mean(dT)+5
    star_offset: float = 5.0
    k_sigma_t4: float = 3.0
    k_sigma_dt: float = 3.5
    min_ddt: float = 6.0
    min_d4: float = 0.0
    t11_test: bool = True
    k_win: int = 7  # window taps (k x k), spacing dil cells
    dil_modis: int = 8  # MODIS native pixels are ~2.7 cells wide: spacing 8 cells ~ 3 pixels, span ~ +-9 pixels (MOD14 uses 21 x 21 pixels)
    dil_viirs: int = 2
    min_bg: int = 8  # minimum background cells for the contextual test


def _box(x, k, dil):
    ones = torch.ones(1, 1, k, k, device=x.device, dtype=x.dtype)
    return F.conv2d(x, ones, padding=dil * (k // 2), dilation=dil)


def detect(L4, L11, valid, day, lam4: float, lam11: float, cfg: RuleConfig | None = None) -> torch.Tensor:
    """L4, L11 [B,H,W] radiance (target band), valid [B,H,W] bool, day [B] bool -> fire [B,H,W] bool (False outside `valid`)."""
    cfg = cfg or RuleConfig()
    modis = abs(lam4 - planck.LAMBDA["modis_4"]) < 1e-6
    dil = cfg.dil_modis if modis else cfg.dil_viirs
    dev = L4.device
    L4 = torch.where(valid, L4, torch.ones_like(L4)).clamp(min=1e-6); L11 = torch.where(valid, L11, torch.ones_like(L11)).clamp(min=1e-6)
    T4, T11 = planck.brightness_temperature(L4, lam4), planck.brightness_temperature(L11, lam11)
    dT = T4 - T11
    d = day[:, None, None]
    pick = lambda pair: torch.where(d, torch.tensor(pair[1], device=dev), torch.tensor(pair[0], device=dev))
    vf = valid.float()[:, None]
    n_all = _box(vf, cfg.k_win, dil).clamp(min=1)
    stat = lambda x, w, n: (_box(x[:, None] * w, cfg.k_win, dil) / n.clamp(min=1))[:, 0]
    if modis and cfg.mode == "c6":
        m4, md = stat(T4, vf, n_all), stat(dT, vf, n_all)
        t4s = (m4 + cfg.star_offset).clamp(*cfg.t4_star); dts = (md + cfg.star_offset).clamp(*cfg.dt_star)
        pot = valid & (T4 > t4s) & (dT > dts)
    elif modis:
        pot = valid & (T4 > pick(cfg.t4_pot)) & (dT > pick(cfg.dt_pot))
    else:
        pot = valid & (T4 > pick(cfg.t4_pot_viirs)) & (dT > pick(cfg.dt_pot))
    bgv = (valid & ~pot).float()[:, None]
    n = _box(bgv, cfg.k_win, dil)
    mu4, mud, mu11 = stat(T4, bgv, n), stat(dT, bgv, n), stat(T11, bgv, n)
    sd_ = lambda x, mu: (stat(x * x, bgv, n) - mu**2).clamp(min=0).sqrt()
    s4, sd, s11 = sd_(T4, mu4), sd_(dT, mud), sd_(T11, mu11)
    ctx = (T4 > mu4 + torch.clamp(cfg.k_sigma_t4 * s4, min=cfg.min_d4)) & (dT > mud + torch.clamp(cfg.k_sigma_dt * sd, min=cfg.min_ddt)) & (n[:, 0] >= cfg.min_bg)
    if cfg.t11_test:
        ctx = ctx & (T11 > mu11 + s11 - 4.0)
    absolute = T4 > pick(cfg.t4_abs if modis else cfg.t4_abs_viirs)
    return valid & pot & (absolute | ctx)
