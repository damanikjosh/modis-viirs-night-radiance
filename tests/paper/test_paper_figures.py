"""Figures and the stored-result experiment (A1-A4) regenerate from results/ and reproduce the recorded A1 numbers."""
import csv, glob, json, os, subprocess, sys

import pytest

from conftest import ROOT

FIG = sorted(glob.glob(os.path.join(ROOT, "scripts", "paper", "figures", "make_*.py")))


def run(args, **kw):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, **kw)
    assert r.returncode == 0, r.stdout[-1000:] + r.stderr[-2000:]


def test_figure_scripts_present():
    assert len(FIG) == 11 and not any("timeline" in f for f in FIG)


@pytest.mark.parametrize("script", FIG, ids=[os.path.basename(f) for f in FIG])
def test_figure_renders(script, tmp_path):
    run([script, "--out", str(tmp_path)])
    out = [f for f in os.listdir(tmp_path) if f.endswith(".pdf")]
    assert out, "no PDF written"
    for f in out:
        assert open(tmp_path / f, "rb").read(5) == b"%PDF-" and os.path.getsize(tmp_path / f) > 5000


def test_a_stored_reproduces_recorded_numbers(tmp_path):
    run([os.path.join(ROOT, "scripts", "exp", "a_stored.py"), os.path.join(ROOT, "results"), str(tmp_path)])
    summ = json.load(open(tmp_path / "a1_seed_spread.json"))["summary"]
    got = {}
    for r in summ:
        if r["checkpoint"] != "best_val":
            continue
        sep = sum(1 for p in r["pairs"] if p["separated"] and "oldloss-nf2" not in (p["a"], p["b"]))
        got[r["quantity"].split(" (")[0]] = (sep, r["gamma_family_pairs"])
    rec = {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(os.path.join(ROOT, "results", "static_numbers.tsv"), encoding="utf8") if l.startswith("ExpSep")}
    assert got["hottest-cell share"] == (rec["ExpSepHot"], rec["ExpSepPairs"])
    assert got["block gain over best reference"] == (rec["ExpSepRank"], rec["ExpSepPairs"])
    assert got["3 km amount error: Oracle-background spread minus model"][0] == rec["ExpSepAmount"]
    assert got["flag count ratio"][0] == rec["ExpSepFlags"]
    assert got["energy ratio, MODIS-selected"][0] == rec["ExpSepEnergy"]
    for n in ("a2_learning_curves.pdf", "a3_small_vs_multi.pdf", "a4_learned_vs_simple.pdf"):
        assert (tmp_path / n).exists()


def test_scripts_compile():
    files = []
    for d in ("paper", "exp", "design"):
        files += glob.glob(os.path.join(ROOT, "scripts", d, "**", "*.py"), recursive=True)
    assert len(files) >= 30
    for f in files:
        compile(open(f).read(), f, "exec")
