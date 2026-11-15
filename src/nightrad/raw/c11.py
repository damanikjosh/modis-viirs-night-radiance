"""MODIS 11 um radiance on the 375 m VIIRS grid, in the VIIRS I5 band, by physics only: no learning, no VIIRS value.

Used by the detection rule when the model predicts the 4 um band alone: the rule needs a 11 um radiance next to the predicted 4 um one, and the MODIS 11 um measurement
(its own input) supplies it. Steps (all fixed):
  1. placement: the tiles already carry, for every 375 m cell, the value of the MODIS native pixel that covers it (modis_pid), with the MODIS validity class in modis_code; a cell
     whose own pixel is invalid (cloud, water, bad, bow-tie, not covered) stays invalid, nothing is filled from a neighbour; the mask comes along (common mask = MODIS and VIIRS valid,
     the mask the model and the loss use);
  2. band conversion: MODIS band 31 radiance -> the radiance a blackbody of the same temperature would give in the VIIRS I5 band (planck.convert_background, the fitted band constants
     of the active planck.use_bands set). 11 um barely responds to sub-pixel fire, so the whole radiance is treated as background-like.
Nothing read here comes from VIIRS values: only modis_L, modis_code (and viirs_code for the common mask, a validity class, not a measurement)."""
from __future__ import annotations

import torch

from nightrad.raw import planck
from nightrad.raw.mask import common_mask


def modis11_on_viirs_grid(inp: dict, widen: int = 0):
    """inp: model-input dict (schema.INPUT_KEYS). -> (L11 [B,H,W] float32, valid [B,H,W] bool): MODIS 11 um in the VIIRS I5 band on every 375 m cell, valid where the common mask holds
    (0 elsewhere)."""
    valid = common_mask(inp["modis_code"], inp["viirs_code"], widen)
    Lm = inp["modis_L"][:, 1].float()
    Lm = torch.where(valid, Lm, torch.ones_like(Lm))  # NaN-safe: nothing outside the mask is ever converted
    L = planck.convert_background(Lm.clamp(min=1e-6), planck.LAMBDA["modis_11"], planck.LAMBDA["viirs_11"])
    return torch.where(valid, L, torch.zeros_like(L)), valid
