#!/bin/bash
# Experiment D: ONE short-budget training run on a pairing-window subset (arm a|b|c; subsets built by d_subsets.py into $NR_OUT/D/subsets/<arm>).
# Same hyper-parameters as the Gamma-BW100 (short) run of experiment C (c_run.sh bw100) except: epochs so that the step count is equal (1,750 steps vs 1,752 at 2,340 tiles) and sparser validation/diagnostics.
# usage: NR_DATA_ROOT=<dir> scripts/exp/d_run.sh <arm> <seed> [epochs]       NOTE: needs train_raw.py with the --val-select option.
source "$(dirname "$0")/env.sh"
arm=$1; seed=$2; ep=${3:-125}; name=dabl-$arm-s$seed; D="$NR_OUT/D/subsets/$arm"
mkdir -p "$NR_OUT/runs"
exec $PYTHON scripts/train_raw.py --data "$D/train_night" --extra-data "$D/nf_train_night" --val-data "$PACKS/val2_night" --val-select configs/val2_split.json --val-side select --region-balance \
  --loss gamma --loss-family gamma --w-energy 0.1 --hide-jitter 3 --lr 2e-3 --lr-const --warmup 100 --bs 32 --epochs "$ep" --amp bf16 --workers 4 --seed "$seed" --no-legacy-val --diag-every 25 --diag-proto-every 25 --diag-tile-every 25 --eval-every 25 \
  --bright-weight 100 --out "$NR_OUT/runs/$name" > "$NR_OUT/runs/$name.log" 2>&1
