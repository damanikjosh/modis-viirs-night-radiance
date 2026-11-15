"""VIIRS I4 saturation: VIIRS I4 saturation is never flagged, and saturated / folded-over pixels carry wrong T4.

Evidence on a real granule VNP14IMG.A2019246.1242 + VNP02IMG: the product itself sets FP_T4 = 367 K on saturated fire pixels
(272 of 8163), but the L1B-LUT based `sat` flag is False for all of them and their T4 reads 356-358 K (260 px) or has
folded over to 208-350 K (12 px). The tests skip when the raw granules are absent (NR_RAW_DIR, default data/raw)."""
import glob
import os
from pathlib import Path

import numpy as np
import pytest

from nightrad.data.readers import Swath, attach_l1b, read_viirs, read_viirs_l1b


@pytest.fixture(scope="session")
def raw():
    p = Path(os.environ.get("NR_RAW_DIR", "data/raw"))
    if not p.exists():
        pytest.skip(f"raw granules not found at {p} (set NR_RAW_DIR)")
    return p


def _one(raw, pattern):
    hits = sorted(glob.glob(str(raw / pattern)))
    if not hits:
        pytest.skip(f"no file {pattern}")
    return hits[0]


def test_saturated_fire_pixels_are_flagged_on_real_granule(raw):
    import netCDF4

    v14 = _one(raw, "VNP14IMG/VNP14IMG.A2019246.1242*")
    v03 = _one(raw, "VNP03MOD/*A2019246.1242*")
    v02 = _one(raw, "VNP02IMG/*A2019246.1242*")
    with netCDF4.Dataset(v14) as d:
        d.set_auto_maskandscale(False)
        fl, fs = d["FP_line"][:].astype(int), d["FP_sample"][:].astype(int)
        satprod = d["FP_T4"][:] >= 366.9  # the product's own saturation marker
    sw = read_viirs(v14, v03)
    attach_l1b(sw, *read_viirs_l1b(v02))
    assert satprod.sum() > 200
    flagged = sw.sat[fl, fs][satprod].mean()
    t4 = sw.t4[fl, fs][satprod]
    assert flagged > 0.95, f"only {flagged:.1%} of product-saturated I4 pixels have sat=True"
    assert np.nanmin(t4) >= 356.0, f"saturated pixel with T4 = {np.nanmin(t4):.1f} K (fold-over)"


def test_attach_l1b_marks_declared_saturation_and_repairs_fold_over():
    z = np.zeros((4, 4), np.float32)
    sw = Swath("viirs", "snpp", "x", z, z, np.zeros((4, 4), np.uint8), z, z, z, 0.0, 360.0)
    prod = np.zeros((4, 4), bool); prod[1, 1] = prod[2, 2] = prod[3, 3] = True
    sw.sat_prod = prod
    t4 = np.full((4, 4), 300.0, np.float32); t4[1, 1] = 357.0; t4[2, 2] = 208.0; t4[3, 3] = np.nan  # plateau, fold-over, missing
    attach_l1b(sw, t4, np.full((4, 4), 290.0, np.float32), np.zeros((4, 4), bool))
    assert sw.sat[1, 1] and sw.sat[2, 2] and not sw.sat[3, 3] and not sw.sat[0, 0]
    assert sw.t4[1, 1] == 367.0 and sw.t4[2, 2] == 367.0 and np.isnan(sw.t4[3, 3]) and sw.t4[0, 0] == 300.0
