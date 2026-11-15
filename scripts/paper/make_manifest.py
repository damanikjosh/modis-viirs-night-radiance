#!/usr/bin/env python3
"""(Re)write results/MANIFEST.json: size and sha256 of every file under results/ (except the manifest itself).

Usage:  python scripts/paper/make_manifest.py [--results results]

Per-file "kind" and "note" fields of an existing manifest are kept. A file that was derived from an original result file (fields not needed for any number were dropped) carries "derived": true and the sha256 of the original in "source_sha256"; set these by editing the manifest
once (they are preserved here). New files get kind "stored result" unless their path says otherwise.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_manifest import sha256

KIND_BY_TOP = {"static_numbers.tsv": "recorded value", "static_tables": "recorded value", "metrics2_summary.json": "compacted summary", "design": "design-test output"}


def build(res_dir, notes=None):
    res_dir = os.path.abspath(res_dir)
    old = {}
    mp = os.path.join(res_dir, "MANIFEST.json")
    if os.path.exists(mp):
        old = {e["local"]: e for e in json.load(open(mp))["files"]}
    files = []
    for dp, _, fs in sorted(os.walk(res_dir)):
        for fn in sorted(fs):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, res_dir)
            if rel == "MANIFEST.json":
                continue
            loc = "results/" + rel
            e = {"local": loc, "size": os.path.getsize(p), "sha256": sha256(p)}
            prev = old.get(loc, {})
            e["kind"] = prev.get("kind") or KIND_BY_TOP.get(rel.split(os.sep)[0], "stored result")
            for k in ("derived", "source_sha256", "note"):
                if k in prev:
                    e[k] = prev[k]
            if notes and loc in notes:
                e.update(notes[loc])
            files.append(e)
    man = {"note": "Stored result files that scripts/paper/* read. Verify with scripts/paper/verify_manifest.py. 'derived' files were derived from the original (fields not needed for any number dropped); source_sha256 is the hash of the original.",
           "n_files": len(files), "files": files}
    json.dump(man, open(mp, "w"), indent=1)
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))
    m = build(ap.parse_args().results)
    print("wrote MANIFEST.json:", m["n_files"], "files")
