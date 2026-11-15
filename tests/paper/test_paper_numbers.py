"""Every number macro and table of the paper regenerates, byte for byte, from the stored result files."""
import glob, json, os, re, shutil, subprocess, sys

import pytest

from conftest import ROOT

EXPECTED = os.path.join(os.path.dirname(__file__), "expected")


def body(path):
    """file content without the generator comment line (the only line that names the generating script)"""
    return [l for l in open(path, encoding="utf8").read().splitlines() if not l.startswith("% GENERATED")]


def test_manifest_matches(tmp_path):
    import verify_manifest
    res = os.path.join(ROOT, "results")
    assert verify_manifest.verify(res) == []
    # a changed file is detected
    cp = tmp_path / "results"
    shutil.copytree(res, cp)
    victim = cp / "raw" / "sweep1" / "rows.jsonl"
    victim.write_text(victim.read_text() + " ")
    assert any("rows.jsonl" in p for p in verify_manifest.verify(str(cp)))
    # an unlisted file is detected
    (cp / "raw" / "extra.json").write_text("{}")
    assert any("extra.json" in p for p in verify_manifest.verify(str(cp)))


def test_generated_equals_paper(gen_out):
    names = sorted(os.listdir(EXPECTED))
    assert "numbers.tex" in names and len(names) == 19
    for n in names:
        assert body(os.path.join(EXPECTED, n)) == body(os.path.join(gen_out, n)), n


def test_all_macros_defined_once(gen_out):
    macros = re.findall(r"\\newcommand\{\\(N[A-Za-z]+)\}", open(os.path.join(gen_out, "numbers.tex")).read())
    assert len(macros) == len(set(macros)) == 1204


def test_number_map_written(gen_out):
    md = open(os.path.join(gen_out, "NUMBERS.md")).read()
    assert md.count("| `\\N") == 1204
    assert "s3://" not in md


def test_closed_test_files_are_aggregates_only():
    """The held-out episode enters only as the pre-stated line verdicts and score aggregates: no per-tile or per-pixel array."""
    d = os.path.join(ROOT, "results", "raw", "_closed_test")
    assert sorted(os.listdir(d)) == ["closed_test_record.json", "score_night.json", "verdict_day.json", "verdict_night.json"]

    def longest(o):
        if isinstance(o, list):
            return max([len(o)] + [longest(x) for x in o])
        if isinstance(o, dict):
            return max([0] + [longest(v) for v in o.values()])
        return 0
    for n in os.listdir(d):
        assert longest(json.load(open(os.path.join(d, n)))) <= 22, n


def test_figure_line_counts_agree_with_generator(gen_out):
    """figures/common.py keeps its own copy of the line logic; it must agree with the generator on the stored files."""
    sys.path.insert(0, os.path.join(ROOT, "scripts", "paper", "figures"))
    import common
    runs, _ = common.load_select()
    macros = dict(re.findall(r"\\newcommand\{\\(N[A-Za-z]+)\}\{([^}]*)\}", open(os.path.join(gen_out, "numbers.tex")).read()))
    grp = common.by_group(runs)
    for g in ("Old", "Gam", "H", "T", "HF", "TF"):
        for k, w in (("L1", "LOne"), ("L2", "LTwo"), ("L3", "LThree"), ("L4", "LFour"), ("L5", "LFive")):
            assert sum(common.lines_of(r["j"])[k] for r in grp[g]) == int(macros[f"N{g}{w}"]), (g, k)


def test_recorded_numbers_that_can_be_cross_checked():
    """Recorded values that a stored file also holds must agree with it."""
    rec = {l.split("\t")[0]: l.split("\t")[1] for l in open(os.path.join(ROOT, "results", "static_numbers.tsv"), encoding="utf8") if l.strip() and not l.startswith("#")}
    log = [json.loads(l) for l in open(os.path.join(ROOT, "results", "raw", "gammabw1000-nf2-s0", "log.jsonl"))]
    assert rec["FullSteps"] == f"{log[-1]['step']:,}"
    assert len(log) == 46
