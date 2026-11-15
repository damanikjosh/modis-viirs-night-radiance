#!/usr/bin/env python
"""Write the frozen run configuration (configs/raw.json): the default RunConfig, the detection-rule constants and the excluded episodes.
A split whose name or recorded split field starts with "test" cannot be opened (prereg.open_split) unless this file exists and the run
configuration hashes to its `run_hash`. No test split is used in the paper; the held-out episode has its own one-time script (closed_test.py).
Changing a default later means: report the change, re-freeze, new hash. configs/raw.json was re-frozen when NetConfig gained `out4` (4 um-only option).
  python scripts/freeze_config.py configs/raw.json"""
import dataclasses
import sys

from nightrad.raw import exclude, prereg
from nightrad.raw.core import RunConfig
from nightrad.raw.rules import RuleConfig

cfg = {
    "run": RunConfig().to_dict(),
    "rule": dataclasses.asdict(RuleConfig()),
    "exclusions": exclude.EPISODES,
    "tuning_data": "train + validation only; a test split needs --allow-test and is logged in test_access.log",
    "input_policy": "VIIRS cells at the 367 K placeholder (written by the tile builder where the product declares saturation or no processing) are removed by the mask; viirs_sat is never an input (zeroed); every radiance and saturation channel is zero outside the mask; modis_sat and band21 come from the L1B radiances",
    "daynight": "night only (cos SZA < 0) by default; day and both are explicit switches",
}
print("frozen hash", prereg.freeze(cfg, sys.argv[1] if len(sys.argv) > 1 else "configs/raw.json"))
