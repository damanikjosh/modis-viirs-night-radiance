#!/bin/bash
# Experiment C: score short-budget ablation checkpoints (last.pt = final step) on SELECT only with the guarded scorer.
# usage: NR_DATA_ROOT=<dir> scripts/exp/c_score.sh <git sha7> <run name> [<run name> ...]
# The input-blind constant only feeds the 'const' simple-method row (value from the big-data TRAIN pack; no model number uses it).
source "$(dirname "$0")/env.sh"
echo '{"const": -0.36296144127845764}' > "$NR_OUT/const.json"
sha=$1; shift
$PYTHON scripts/score_run.py --data "$PACKS/val2_select" --train UNUSED --const-json "$NR_OUT/const.json" --set select --runs-dir "$NR_OUT/runs" --names "$@" --ckpt last.pt --out "$NR_OUT/scores" --git-sha "$sha" --split configs/nf2_split_episodes.json --force
