import math
import numpy as np
from viirs_footprint import footprint_km, scan_angle_deg, R, H, T_MAX


def test_nadir_is_375_m():
    s, t, c = footprint_km([0.0]); assert abs(s[0] - 0.375) < 1e-9 and abs(t[0] - 0.375) < 1e-9 and not c[0]


def test_edge_of_scan_about_800_m():   # published: ~800 m (NASA VIIRS slides) / 790 m (Wolfe 2013) at the end of the scan
    z = math.degrees(math.asin((R + H) / R * math.sin(math.radians(T_MAX))))
    s, t, c = footprint_km([z - 1e-6]); assert 0.76 <= s[0] <= 0.84 and 0.76 <= t[0] <= 0.84 and not c[0]


def test_swath_half_width_1520_km():   # published swath 3,040 km: ground distance = R * (z - t)
    z = math.radians(math.degrees(math.asin((R + H) / R * math.sin(math.radians(T_MAX)))))
    assert abs(R * (z - math.radians(T_MAX)) - 1520) < 40


def test_aggregation_zone_sawtooth_and_clamp():
    z = lambda t: math.degrees(math.asin((R + H) / R * math.sin(math.radians(t))))
    a = footprint_km([z(31.58), z(31.60), z(44.67), z(44.69)])[0]
    assert a[1] < a[0] and a[3] < a[2]                      # size drops where fewer-aggregated zones start
    s, t, c = footprint_km([72.0, 65.5]); assert c[0] and not c[1] and abs(s[0] - footprint_km([69.9])[0][0]) < 0.02
    assert np.all(np.diff(footprint_km(np.linspace(0, 31, 20))[1]) > 0)   # track grows with angle
