#!/bin/bash
# Experiment C: ONE short-budget ablation training run on the small list (train_night + nf_train_night, 2,340 tiles). NOT converged: fixed step budget, final checkpoint only.
# usage: NR_DATA_ROOT=<dir> scripts/exp/c_run.sh <config> <seed> [epochs]
#   epochs of 2,048 samples = 64 steps each (default 24 = 1,536 steps; the old-loss/gamma small-list runs reached best validation at ~1,500 steps)
# configs: base | bw100 | bw1000 | dmax0 | noenergy | novvmm   (everything else identical: gamma loss, region balance, bs 32, lr 2e-3 const, warmup 100, bf16, hide-jitter 3, no early stop)
# NOTE: needs train_raw.py with the --val-select option (stage-0 validation split configs/val2_split.json).
source "$(dirname "$0")/env.sh"
cfg=$1; seed=$2; ep=${3:-24}; name=abl-$cfg-s$seed
extra=""
case $cfg in
  base) ;; bw100) extra="--bright-weight 100";; bw1000) extra="--bright-weight 1000";;
  dmax0) extra="--delta-max 0";; noenergy) extra="--w-energy 0";; novvmm) extra="--w-vv 0 --w-mm 0";;
  *) echo "unknown config"; exit 2;;
esac
WE="--w-energy 0.1"; [ "$cfg" = noenergy ] && WE=""
mkdir -p "$NR_OUT/runs"
exec $PYTHON scripts/train_raw.py --data "$PACKS/train_night" --extra-data "$PACKS/nf_train_night" --val-data "$PACKS/val2_night" --val-select configs/val2_split.json --val-side select --region-balance \
  --loss gamma --loss-family gamma $WE --hide-jitter 3 --lr 2e-3 --lr-const --warmup 100 --bs 32 --epochs "$ep" --amp bf16 --workers 4 --seed "$seed" --no-legacy-val --diag-every 8 --eval-every 4 \
  $extra --out "$NR_OUT/runs/$name" > "$NR_OUT/runs/$name.log" 2>&1
