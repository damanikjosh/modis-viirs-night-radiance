#!/bin/bash
# Experiment C: run the ablation configurations one after the other (two seeds each, trained in parallel). A gated configuration (dmax0|noenergy|novvmm) is trained only if its four sanity gates pass.
# usage: NR_DATA_ROOT=<dir> scripts/exp/c_queue.sh <config> [<config> ...]
source "$(dirname "$0")/env.sh"
for cfg in "$@"; do
  case $cfg in dmax0|noenergy|novvmm)
    bash scripts/exp/c_gates.sh "$cfg"
    $PYTHON scripts/exp/c_gate_eval.py "$NR_OUT/gates/$cfg" > "$NR_OUT/queue_$cfg.verdict" 2>&1 || { echo "gates FAIL, not training $cfg" >> "$NR_OUT/queue_$cfg.verdict"; continue; };;
  esac
  for s in 0 1; do bash scripts/exp/c_run.sh "$cfg" "$s" & done
  wait
done
