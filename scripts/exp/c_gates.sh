#!/bin/bash
# Experiment C: the four sanity gates (path 50 / seed 40 / share / overfit 10k steps) for one ablation configuration, on the small-list train pack.
# usage: NR_DATA_ROOT=<dir> scripts/exp/c_gates.sh <config>      then: python scripts/exp/c_gate_eval.py $NR_OUT/gates/<config>
source "$(dirname "$0")/env.sh"
cfg=$1; G="$NR_OUT/gates/$cfg"; mkdir -p "$G"
extra=""; WE="--w-energy 0.1"
case $cfg in dmax0) extra="--delta-max 0";; noenergy) WE="";; novvmm) extra="--w-vv 0 --w-mm 0";; bw100) extra="--bright-weight 100";; bw1000) extra="--bright-weight 1000";; base) ;; *) exit 2;; esac
com="--data $PACKS/train_night --loss gamma --loss-family gamma $WE --hide-jitter 3 --lr-const --warmup 100 --workers 0 $extra"
for g in "path 50" "seed 40" "share 0" "overfit 10000"; do set -- $g
  if [ "$1" = share ]; then st=""; else st="--steps $2"; fi
  $PYTHON scripts/diag_gates.py --gate "$1" $st $com --out "$G/g_$1" --out-json "$G/$1.json" > "$G/$1.log" 2>&1
  tail -1 "$G/$1.log" | cut -c1-300
done
