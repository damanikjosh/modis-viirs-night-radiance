"""Version rules of the layer product.

PIPELINE_VERSION is written into every per-scene record and parquet header by the worker. Bump it for every change that can alter the flagged cells
(tiling, model call, flag rule, per-cell columns). The aggregate step only accepts scenes whose version is >= its `--min-version`; a record without a
version counts as "v1.0". History: v1.0 first run; v1.1 OOM retry + size-checked downloads; v1.2 overlapping tiles with interior crop (margin 12 cells),
tile rule `modis_firelevel`, extra per-cell columns, bounding-box clip.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

PIPELINE_VERSION = "v1.2"
CKPT_SHA256 = "87251b2bb8f151d3e941922862f5f146d7b6e0dff02f519d337ab6e8e523dc26"
CKPT_EPOCH = 17            # epochs trained when the checkpoint was written (1-based)
DEFAULT_CKPT = Path(__file__).resolve().parents[3] / "weights" / "best_val.pt"


def vkey(v) -> tuple[int, int]:
    """'v1.2' -> (1, 2); a missing version is v1.0; anything unparsable is (0, 0)."""
    m = re.match(r"v?(\d+)\.(\d+)", str(v or "v1.0"))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_checkpoint(path) -> str:
    """Refuse any checkpoint other than the one the layer was built with."""
    got = sha256_file(path)
    if got != CKPT_SHA256:
        raise SystemExit(f"checkpoint sha256 mismatch: {path} is {got}, expected {CKPT_SHA256}")
    return got
