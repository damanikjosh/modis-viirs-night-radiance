"""guard review: end-to-end wiring of the REPORT guard in scripts/score_run.py with synthetic packs (the scorer is run as a subprocess; it must stop at the guard BEFORE loading any tile)."""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT = os.path.join(ROOT, "configs", "nf2_split.json")


def make_pack(d, names, episodes, with_names=True, names_len=None, n_override=None):
    """Synthetic pack: meta.json, eval_episode.npy, episodes.json (+ names.json). No tile data (the guard must act before any tile is read)."""
    os.makedirs(d, exist_ok=True); n = len(names)
    json.dump({"n": n_override or n, "keys": ["eval_episode"]}, open(os.path.join(d, "meta.json"), "w"))
    ep = {e: i for i, e in enumerate(sorted(set(episodes)))}
    np.save(os.path.join(d, "eval_episode.npy"), np.array([ep[e] for e in episodes], np.int32)); json.dump(ep, open(os.path.join(d, "episodes.json"), "w"))
    if with_names:
        json.dump(names if names_len is None else (names + ["x"] * (names_len - len(names)))[:names_len], open(os.path.join(d, "names.json"), "w"))
    return str(d)


def run(data, set_, split_eps, extra=(), nf2_split=SPLIT, tmp=None):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "score_run.py"), "--data", data, "--train", data, "--runs-dir", str(tmp), "--out", str(tmp / "out"), "--set", set_, "--git-sha", "abcdef1234",
           "--split", split_eps, "--nf2-split", nf2_split, "--names", "none", *extra]
    return subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}, capture_output=True, text=True, timeout=300)


def lists():
    t = json.load(open(SPLIT))["tiles"]
    return t["select"], t["report"]


def eps_file(tmp):
    f = tmp / "eps.json"; json.dump({"select": ["ES"], "report": ["ER"]}, open(f, "w")); return str(f)


def test_report_only_pack_is_refused(tmp_path):
    sel, rep = lists()
    d = make_pack(tmp_path / "p", rep[:5], ["ER"] * 5)
    r = run(d, "val2all", eps_file(tmp_path), tmp=tmp_path)
    assert r.returncode != 0 and "GUARD" in r.stderr + r.stdout and "REPORT list" in r.stderr + r.stdout


def test_mixed_pack_scored_whole_is_refused(tmp_path):
    sel, rep = lists()
    d = make_pack(tmp_path / "p", sel[:3] + rep[:2], ["ES"] * 3 + ["ER"] * 2)
    r = run(d, "val2all", eps_file(tmp_path), tmp=tmp_path)
    assert r.returncode != 0 and "REPORT list" in r.stderr


def test_select_subset_of_a_mixed_pack_passes_the_guard(tmp_path):
    sel, rep = lists()
    d = make_pack(tmp_path / "p", sel[:3] + rep[:2], ["ES"] * 3 + ["ER"] * 2)
    r = run(d, "select", eps_file(tmp_path), tmp=tmp_path)
    assert "GUARD {" in r.stdout and "'checked': True" in r.stdout and "'tiles_checked': 3" in r.stdout and "'report_hits': 0" in r.stdout and "'split_pinned': True" in r.stdout


def test_pack_without_ids_is_refused_unless_a_reason_is_given(tmp_path):
    sel, rep = lists()
    d = make_pack(tmp_path / "p", sel[:3], ["ES"] * 3, with_names=False)
    r = run(d, "val2all", eps_file(tmp_path), tmp=tmp_path)
    assert r.returncode != 0 and "no names.json" in r.stderr
    r2 = run(d, "val2all", eps_file(tmp_path), extra=["--allow-unnamed-reason", "synthetic"], tmp=tmp_path)
    assert "'checked': False" in r2.stdout


def test_the_old_589_tile_cross_data_list_is_refused(tmp_path):
    x = json.load(open(os.path.join(ROOT, "configs", "xdata_list.json")))["tiles"]
    assert len(x) == 589
    d = make_pack(tmp_path / "p", x, ["ES"] * 589)
    r = run(d, "val2all", eps_file(tmp_path), tmp=tmp_path)
    assert r.returncode != 0 and "88 RETIRED tiles" in r.stderr


def test_ids_file_length_must_equal_pack_size(tmp_path):
    sel, rep = lists()
    d = make_pack(tmp_path / "p", sel[:3], ["ES"] * 3, names_len=5)     # a longer ids file would pass silently without the check
    r = run(d, "select", eps_file(tmp_path), tmp=tmp_path)
    assert r.returncode != 0 and "ids but the pack holds" in r.stderr


def test_other_split_file_is_refused_without_a_logged_override(tmp_path):
    sel, rep = lists()
    d = make_pack(tmp_path / "p", sel[:3], ["ES"] * 3)
    other = tmp_path / "other.json"; json.dump({"tiles": {"train": [], "select": sel, "report": []}}, open(other, "w"))   # a split file with an EMPTY report list would switch the guard off
    r = run(d, "select", eps_file(tmp_path), nf2_split=str(other), tmp=tmp_path)
    assert r.returncode != 0 and "not the committed" in r.stderr
    r2 = run(d, "select", eps_file(tmp_path), nf2_split=str(other), extra=["--split-override-reason", "unit test"], tmp=tmp_path)
    assert "'split_override_reason': 'unit test'" in r2.stdout and "'split_pinned': False" in r2.stdout
