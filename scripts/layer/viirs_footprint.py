"""Nominal VIIRS I-band (375 m) pixel footprint as a function of the viewing angle (display convention; NOT a measurement).
Geometry (S-NPP, altitude H = 829 km, spherical Earth R = 6371 km); local zenith angle z (our `vza`) <-> VIIRS scan angle t:  sin z = (R + H) / R * sin t.
Native I-band footprint at nadir: 375 m x 375 m (Wolfe et al. 2013, JGR Atmos 118, 11508, "SNPP VIIRS spatial resolution and geometric characterization": I-band HSI 375 m = half of the M-band 750 m).
Growth with angle (slant range s = (R + H) cos t - R cos z):  along-track = 375 m * s / H ;  along-scan = 375 m * s / (H cos z) * (n / 3), where n = samples aggregated across the scan:
3 for |t| < 31.59 deg, 2 for 31.59-44.68 deg, 1 for 44.68-56.28 deg (the three aggregation zones; NOAA VIIRS SDR data format / Wolfe 2013 Fig. 3: ~32 and ~45 deg).
Beyond the maximum scan angle 56.28 deg (z = ~69.9 deg) the edge value is used and counted (`clamped`)."""
import numpy as np

R, H = 6371.0, 829.0
NADIR_KM = 0.375
ZONES = ((31.59, 3), (44.68, 2), (56.28, 1))     # (upper scan angle deg, samples aggregated across scan)
T_MAX = 56.28


def scan_angle_deg(vza_deg):
    s = np.clip(R / (R + H) * np.sin(np.radians(np.asarray(vza_deg, dtype=float))), 0, 1)
    return np.degrees(np.arcsin(s))


def footprint_km(vza_deg):
    """-> (scan_km, track_km, clamped) arrays for local zenith angles in degrees."""
    z = np.asarray(vza_deg, dtype=float)
    t = scan_angle_deg(z); clamped = t > T_MAX; t = np.minimum(t, T_MAX)
    zr = np.arcsin(np.clip((R + H) / R * np.sin(np.radians(t)), 0, 1)); tr = np.radians(t)
    slant = (R + H) * np.cos(tr) - R * np.cos(zr)
    n = np.where(t < ZONES[0][0], ZONES[0][1], np.where(t < ZONES[1][0], ZONES[1][1], ZONES[2][1]))
    track = NADIR_KM * slant / H
    scan = NADIR_KM * slant / (H * np.cos(zr)) * (n / 3.0)
    return scan, track, clamped
