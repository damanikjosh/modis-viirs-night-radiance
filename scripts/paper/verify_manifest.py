#!/usr/bin/env python3
"""Check the stored result files against results/MANIFEST.json (sha256 and size of every listed file; no unlisted file under results/).

Usage:  python scripts/paper/verify_manifest.py [--results results]     exit 0 = every file matches.

Entries marked "derived" are stored files that were derived from the original result file (fields not needed for any number were dropped, nothing else changed); their "source_sha256" is the hash of the unreduced original and is recorded for provenance only.
"""
import argparse, hashlib, json, os, sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(res_dir):
    """Return a list of problems (empty = ok)."""
    man = json.load(open(os.path.join(res_dir, "MANIFEST.json")))
    bad, listed = [], set()
    for e in man["files"]:
        rel = e["local"]
        listed.add(rel)
        p = os.path.join(res_dir, rel[len("results/"):])
        if not os.path.exists(p):
            bad.append(f"missing {rel}")
        elif os.path.getsize(p) != e["size"]:
            bad.append(f"size differs {rel}: {os.path.getsize(p)} != {e['size']}")
        elif sha256(p) != e["sha256"]:
            bad.append(f"sha256 differs {rel}")
    for dp, _, fs in os.walk(res_dir):
        for fn in fs:
            rel = "results/" + os.path.relpath(os.path.join(dp, fn), res_dir)
            if rel not in listed and rel != "results/MANIFEST.json":
                bad.append(f"not in manifest {rel}")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))
    a = ap.parse_args()
    res = os.path.abspath(a.results)
    problems = verify(res)
    n = len(json.load(open(os.path.join(res, "MANIFEST.json")))["files"])
    if problems:
        print("\n".join(problems))
        sys.exit(f"MANIFEST check FAILED ({len(problems)} problems)")
    print(f"MANIFEST ok: {n} files match sha256 and size")
