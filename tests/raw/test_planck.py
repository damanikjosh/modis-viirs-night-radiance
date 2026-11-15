import math

import pytest
import torch

from nightrad.raw import planck as P

C1, C2 = 1.191042972e8, 14387.76877


def hand(T, lam):  # independent re-typing with math, python floats
    return C1 / (lam**5 * (math.exp(C2 / (lam * T)) - 1))


@pytest.mark.parametrize("lam", [3.74, 3.96, 11.0, 11.45])
@pytest.mark.parametrize("T", [250.0, 300.0, 367.0, 800.0])
def test_radiance_matches_hand_computation(lam, T):
    assert float(P.planck_radiance(T, lam)) == pytest.approx(hand(T, lam), rel=1e-6)


def test_known_values():
    # hand-computed anchors: ceilings and B(3.74 um, 300 K)
    assert hand(300, 3.74) == pytest.approx(0.43900, rel=1e-3)
    s = P.saturation_radiance()
    assert s["viirs_4"] == pytest.approx(4.56, rel=2e-3)
    assert s["modis_4"] == pytest.approx(2.09, rel=3e-3)
    assert s["modis_4_b21"] == pytest.approx(85.5, rel=3e-3)


def test_round_trip_bt():
    T = torch.tensor([200.0, 300.0, 500.0, 1000.0], dtype=torch.float64)
    for lam in (3.74, 11.45):
        assert torch.allclose(P.brightness_temperature(P.planck_radiance(T, lam), lam), T, rtol=1e-10)


def test_background_band_conversion_matches_two_step():
    L = torch.tensor([0.3, 0.44, 1.2], dtype=torch.float64)
    T = P.brightness_temperature(L, 3.74)
    assert torch.allclose(P.convert_background(L, 3.74, 3.96), P.planck_radiance(T, 3.96), rtol=1e-9)
    # plan numbers: MODIS-b22 / VIIRS-I4 radiance ratio 1.53 at 300 K, 11 um ratio 1.027 at 300 K
    assert float(P.convert_background(P.planck_radiance(300.0, 3.74), 3.74, 3.96) / P.planck_radiance(300.0, 3.74)) == pytest.approx(1.53, abs=0.01)
    assert float(P.planck_radiance(300.0, 11.0) / P.planck_radiance(300.0, 11.45)) == pytest.approx(1.027, abs=0.002)


def test_excess_factor_at_fire_temperature():
    f = P.excess_factor(3.74, 3.96, 800.0)
    assert f == pytest.approx(hand(800, 3.96) / hand(800, 3.74), rel=1e-12)
    assert f == pytest.approx(0.98, abs=0.01)  # plan: 0.98 at 800 K
    assert P.excess_factor(3.74, 3.96, 1000.0) == pytest.approx(0.94, abs=0.01)
    assert P.excess_factor(3.74, 3.74, 800.0) == pytest.approx(1.0)


def test_use_bands_switches_in_place_and_back():
    from nightrad.raw import planck
    c = dict(planck.LAMBDA)
    try:
        planck.use_bands("data")
        assert planck.LAMBDA["viirs_4"] == 3.7594 and planck.BANDS_NAME == "data" and planck.LAMBDA["modis_4"] == 3.9714
        # round trip still exact with the data wavelengths
        T = torch.tensor([280.0, 320.0, 400.0])
        assert torch.allclose(planck.brightness_temperature(planck.planck_radiance(T, planck.LAMBDA["viirs_4"]), planck.LAMBDA["viirs_4"]), T, rtol=1e-4)
    finally:
        planck.use_bands("centre")
    assert planck.LAMBDA == c and planck.BANDS_NAME == "centre"
