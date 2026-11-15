"""Night-fire build fixes (night-fire build): F1 VIIRS product QA bit 3 only counts on fire-class pixels; F3 MODIS sat needs a real saturation."""
import numpy as np
from nightrad.data.readers import MODIS_EMISSIVE_INDEX, modis_l1b_full, viirs_saturation_from_product


def test_f1_qa_bit3_only_on_fire_pixels():
    qa = np.full((2, 4), 8, np.uint32)            # bit 3 set everywhere
    raw = np.array([[0, 1, 7, 9], [5, 8, 3, 0]], np.uint8)
    sat = viirs_saturation_from_product(np.zeros(0, int), np.zeros(0, int), np.zeros(0, np.float32), qa, raw)
    assert sat.tolist() == [[False, False, True, True], [False, True, False, False]]
    assert viirs_saturation_from_product(np.zeros(0, int), np.zeros(0, int), np.zeros(0, np.float32), qa).all()  # old behaviour without raw


def test_f3_modis_sat_not_set_when_band22_merely_invalid():
    ev = np.full((16, 1, 3), 1000, np.uint16)
    i22 = MODIS_EMISSIVE_INDEX[22]
    ev[i22, 0] = [65535, 65533, 1000]            # fill, saturated flag, normal
    d = modis_l1b_full(ev, np.full(16, 1e-3), np.zeros(16))
    assert d["sat"][0].tolist() == [False, True, False]
