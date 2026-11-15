"""MODIS-only Swath from MYD021KM + MYD03. NO fire product (MYD14) is read.

The tiler (tiles_gpu) takes the cell validity classes from the `raw` mask of the MYD14 product when one is present (cloud / water / clear land / bad), never its fire classes.
Here the same classes are rebuilt from the L1B + geolocation only, with documented rules:
  water  = MYD03 Land/SeaMask not land (values 1 land and 2 coastline are land; MOD14 takes its water class from the same MOD03 mask)
  cloud  = night cloud test of the MOD14 algorithm (Giglio et al. 2003/2016): 12 um brightness temperature (band 32) < 265 K
  bad    = L1B fill / missing 4 or 11 um (handled by Swath.pixel_vclass)
  clear  = everything else (raw code 5: land, no fire information)
Pseudo mask codes follow the MYD14 'fire mask' values used by readers._VC: 3 water, 4 cloud, 5 non-fire land; 0 = not processed (bad).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nightrad.data.readers import (CONF_OF_CLASS, MODIS_EMISSIVE_INDEX, Swath, _modis_ev, _sds, attach_l1b, granule_start, map_fire_mask,
                                   modis_l1b_full, planck_bt)

BAND32_INDEX = 11  # position of band 32 in EV_1KM_Emissive (bands 20,21,22,23,24,25,27,28,29,30,31,32,...)
BAND32_NU = 832.4  # cm-1, MODIS band 32 effective central wavenumber (MODIS L1B documentation)
CLOUD_T12_K = 265.0  # MOD14 night cloud test, 12 um BT below this = cloud
LAND_VALUES = (1, 2)  # MYD03 Land/SeaMask: 1 land, 2 coastline


def read_modis_production(myd021km, myd03, myd14, duration_s: float = 300.0) -> Swath:
    """PRODUCTION path: exactly the swath the training/episode tiler used: MYD14 'fire mask' supplies the validity classes
    (cloud / water / bad / clear land; fire classes 7-9 collapse to 'clear' in the validity class and live only under eval_* names), L1B the radiances.
    MYD14 is used only for its cloud/water/quality classes, as in training; its fire detections are not an input."""
    from pyhdf.SD import SD, SDC

    from nightrad.data.readers import read_modis

    sw = read_modis(myd14, myd03, duration_s)
    attach_l1b(sw, **modis_l1b_full(*_modis_ev(myd021km)))
    f03 = SD(str(myd03), SDC.READ)
    try:
        sw.landsea = np.asarray(f03.select("Land/SeaMask").get(), np.uint8)
    finally:
        f03.end()
    sw.granule = Path(myd021km).name
    return sw


def modis_l1b_full_from(ev, scales, offsets) -> dict:
    return modis_l1b_full(ev, scales, offsets)
