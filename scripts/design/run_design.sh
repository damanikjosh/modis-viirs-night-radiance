#!/bin/bash
# Design tests, data collection on a CPU machine: TRAIN extraction, SELECT pack, collection (baselines on TRAIN+SELECT, model predictions on SELECT).
# usage: NR_DATA_ROOT=<dir> NR_CKPT=<dir with <run>__best_val.pt and train_const_<run>.json> scripts/design/run_design.sh
# Layout under NR_DATA_ROOT: shards/train-nf-*.tar, select.tar (SELECT-only pack), manifest.parquet.  Output: $NR_OUT/design/collect_*.npz
set -euo pipefail
: "${NR_DATA_ROOT:?set NR_DATA_ROOT}"; : "${NR_CKPT:?set NR_CKPT}"
cd "$(dirname "$0")/../.."; export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3}"; D="$NR_DATA_ROOT"; O="${NR_OUT:-$D/exp}/design"; mkdir -p "$O"
[ -d "$D/select_pack" ] || { mkdir -p "$D/select_pack" && tar -xf "$D/select.tar" -C "$D/select_pack"; }
P=$(dirname "$(find "$D/select_pack" -name meta.json | head -1)")
[ -d "$D/train_npz" ] || $PY scripts/design/extract_train.py "$D/train_npz" "$D"/shards/train-nf-*.tar
[ -f "$O/collect_base.npz" ] || $PY scripts/design/collect.py --out "$O/collect_base.npz" --manifest "$D/manifest.parquet" --train-dir "$D/train_npz" --select-pack "$P" --threads 4
for r in gammabw1000-nf2-s0 gammabw1000-nf2-s1 gammabw100-nf2-s0 gammabw100-nf2-s1; do
  case $r in gammabw100-*) C=gammabw100-nf2-s0;; *) C=gammabw1000-nf2-s0;; esac
  [ -f "$O/collect_sel_$r.npz" ] || $PY scripts/design/collect.py --out "$O/collect_sel_$r.npz" --manifest "$D/manifest.parquet" --select-pack "$P" --ckpt "$NR_CKPT/${r}__best_val.pt" --const "$NR_CKPT/train_const_$C.json" --threads 4
done
# then: python scripts/design/analyze_dt.py --base $O/collect_base.npz --model gammabw1000-s0=$O/collect_sel_gammabw1000-nf2-s0.npz ... --out $O/an
