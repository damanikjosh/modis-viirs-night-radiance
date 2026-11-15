"""analyze_dt.py on a synthetic collect file with a PLANTED time-gap effect: the trend coefficient must recover it, and a null file must give a range containing 0."""
import json, os, subprocess, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def synth(path, slope, n_pairs=120, per=40, seed=1):
    rng = np.random.default_rng(seed); rows = []
    for p in range(n_pairs):
        dt = rng.uniform(-30, 30)
        for i in range(per):
            meas = np.exp(rng.normal(0, 1)) * 0.01; err = rng.normal(0, 0.8) + slope * abs(dt) / 10
            rows.append((p, dt, meas, meas * np.exp(err), p * 3 + i % 3, rng.uniform(0, 55), p % 3))
    r = np.array(rows); n = len(r); side = np.where(r[:, 0] % 4 == 0, "select", "train")
    px = {"pair": r[:, 0].astype(int), "tile": r[:, 4].astype(int), "dt": r[:, 1], "rho4": np.ones(n), "vza": r[:, 5], "psize_scan": np.ones(n), "mexc": r[:, 2] * 3, "meas": r[:, 2], "ncell": np.full(n, 9.0),
          "nvfire": (r[:, 2] > 0.004).astype(float), "pred_uniform": r[:, 3], "pred_excess_only": r[:, 3], "tile_id": np.array([f"t{int(x)}" for x in r[:, 4]]), "side": side, "region": np.array([f"R{int(x)}" for x in r[:, 6]])}
    tl = {"tile_id": px["tile_id"], "pair": px["pair"], "dt": px["dt"], "n_modis_fire": np.ones(n, int), "n_modis_fire_with_v": np.ones(n, int), "n_vfire_fp": np.full(n, 3), "n_vfire_fp_modis_fire": np.ones(n, int), "side": side, "region": px["region"]}
    np.savez_compressed(path, **{"px_" + k: v for k, v in px.items()}, **{"tile_" + k: v for k, v in tl.items()})


def run(tmp, slope):
    f = str(tmp / f"s{slope}.npz"); synth(f, slope); out = str(tmp / f"o{slope}")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "design", "analyze_dt.py"), "--base", f, "--out", out, "--nboot", "60"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.load(open(out + "/summary.json"))


def test_planted_slope_recovered(tmp_path):
    s = run(tmp_path, 0.2)["trend_even_abs_err"]
    # abs of a N(slope*|dt|/10, 0.8) error is not linear in the slope, but the planted effect must be clearly positive and the null file must not be
    assert s["lo"] > 0
    n = run(tmp_path, 0.0)["trend_even_abs_err"]
    assert n["lo"] < 0 < n["hi"]
