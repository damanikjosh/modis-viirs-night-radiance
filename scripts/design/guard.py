"""Design-decision tests: id guard. Only TRAIN and SELECT tiles of configs/nf2_split.json may be read; REPORT and the 88 retired tiles are refused.
Reuses nightrad.raw.nf2_split (pinned sha256 of the split and of the retire list)."""
from __future__ import annotations
import json
from nightrad.raw import nf2_split as NS

ALLOWED_SIDES = ("train", "select")


def allowed_ids(split_json="configs/nf2_split.json", retired_json=None, sides=ALLOWED_SIDES) -> dict[str, str]:
    """{tile id: side} for the allowed sides. Raises if the split file is not the committed one, a side is not allowed, or the result touches REPORT / retired ids."""
    for s in sides:
        if s not in ALLOWED_SIDES:
            raise SystemExit(f"GUARD: side {s!r} is not readable by the analyses (allowed: {ALLOWED_SIDES})")
    retired_json = retired_json or NS.DEFAULT_RETIRED
    if NS.file_sha256(split_json) != NS.COMMITTED_SPLIT_SHA256:
        raise SystemExit("GUARD: split file is not the committed configs/nf2_split.json")
    if NS.file_sha256(retired_json) != NS.COMMITTED_RETIRED_SHA256:
        raise SystemExit("GUARD: retire list is not the committed configs/report_retired.json")
    tiles = json.load(open(split_json))["tiles"]
    out = {t: s for s in sides for t in tiles[s]}
    bad = set(out) & NS.forbidden_tiles(split_json, allow_report=False, retired_json=retired_json)
    if bad:
        raise SystemExit(f"GUARD: {len(bad)} allowed ids are REPORT/retired (e.g. {sorted(bad)[:3]})")
    return out


def check_ids(names, allowed: dict[str, str]):
    """Refuse any tile id outside TRAIN+SELECT (REPORT, retired or unknown)."""
    bad = [n for n in names if n not in allowed]
    if bad:
        raise SystemExit(f"GUARD: {len(bad)} tile ids are not TRAIN/SELECT (e.g. {bad[:3]})")
