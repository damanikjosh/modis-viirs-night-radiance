"""Checkpoint -> predicted VIIRS-view radiance for MODIS(-and-optionally-VIIRS) tiles. Deterministic, no hiding, no fire-product input; fire flags are
the documented rule (nightrad.raw.rules.detect) applied AFTER the model to predicted radiance."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from nightrad.raw import planck
from nightrad.raw.core import Prep, RunConfig, run_scenes
from nightrad.raw.net import RawHarmonizer
from nightrad.raw.rules import detect
from nightrad.raw.schema import split_batch
from nightrad.raw.sweep_eval import ref_cfg


def load_model(ckpt: str, device, daynight: str = "night"):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    planck.use_bands(ck.get("bands", "centre"))
    cfg_run = RunConfig.from_dict(ck["cfg"])
    model = RawHarmonizer(cfg_run.net).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    cfg = ref_cfg(cfg_run)  # same fixed reference settings as protocol_eval_v2 (night, min 10 % common mask)
    cfg.hide = replace(cfg.hide, enabled=False)
    cfg.daynight = daynight
    return model, cfg, cfg_run, {"epoch": ck.get("epoch", -1) + 1, "bands": ck.get("bands", "centre")}


def modis_only_inputs(inp: dict) -> dict:
    """No VIIRS information may enter: the VIIRS side is replaced by a copy of the MODIS validity/geometry (so the common mask is the MODIS mask) and zero radiance."""
    o = dict(inp)
    o["viirs_code"] = inp["modis_code"].clone()
    o["viirs_L"] = torch.zeros_like(inp["viirs_L"])
    o["viirs_sat"] = torch.zeros_like(inp["viirs_sat"]); o["viirs_band21"] = torch.zeros_like(inp["viirs_band21"])
    for k in ("vza", "cossza", "psize"):
        o[f"viirs_{k}"] = inp[f"modis_{k}"].clone()
    return o


@torch.no_grad()
def predict(model, cfg, tiles: dict, device, modis_only: bool = False, bs: int = 8) -> dict:
    """tiles: stacked dict. -> numpy arrays [N,H,W] (float32 / bool): valid, day[N], modis4/11 (own band radiance), pred4/11 (VIIRS band = background + excess),
    bg4/11, exc4/exc11, meas4/11 (NaN without measured VIIRS), flag_model, flag_modis, flag_meas."""
    n = tiles["modis_L"].shape[0]
    acc: dict[str, list] = {}
    L = planck.LAMBDA
    for i0 in range(0, n, bs):
        idx = np.arange(i0, min(n, i0 + bs))
        batch = {k: v[idx] for k, v in tiles.items()}
        inp, _ = split_batch(batch, device)
        inp = modis_only_inputs(inp) if modis_only else inp
        P = Prep(inp, cfg)
        sc, _ = run_scenes(model, P, cfg, None, None)
        s = sc[("modis", "clean")]
        pred = s["B"] + s["E"]
        Lm = inp["modis_L"].float()
        day = P.day
        flag_model = detect(pred[:, 0], pred[:, 1], P.common, day, L["viirs_4"], L["viirs_11"])
        flag_modis = detect(Lm[:, 0], Lm[:, 1], P.common, day, L["modis_4"], L["modis_11"])
        nan = torch.full_like(pred[:, 0], float("nan"))
        if modis_only:
            meas4 = meas11 = nan; flag_meas = torch.zeros_like(P.common)
        else:
            Lv = inp["viirs_L"].float()
            meas4, meas11 = torch.where(P.common, Lv[:, 0], nan), torch.where(P.common, Lv[:, 1], nan)
            flag_meas = detect(Lv[:, 0], Lv[:, 1], P.common, day, L["viirs_4"], L["viirs_11"])
        m = P.common
        put = lambda k, v: acc.setdefault(k, []).append(v.detach().cpu().numpy())
        put("valid", m); put("day", day)
        for name, t in (("modis4", Lm[:, 0]), ("modis11", Lm[:, 1]), ("pred4", pred[:, 0]), ("pred11", pred[:, 1]), ("bg4", s["B"][:, 0]), ("bg11", s["B"][:, 1]),
                        ("exc4", s["E"][:, 0]), ("exc11", s["E"][:, 1])):
            put(name, torch.where(m, t, nan))
        put("meas4", meas4); put("meas11", meas11)
        put("flag_model", flag_model); put("flag_modis", flag_modis); put("flag_meas", flag_meas)
    return {k: np.concatenate(v) for k, v in acc.items()}
