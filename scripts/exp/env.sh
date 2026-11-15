# Shared settings of the experiment runners (sourced). Data lives under NR_DATA_ROOT, outputs under NR_OUT; nothing else is assumed.
#   $NR_DATA_ROOT/packs/{train_night,nf_train_night,val2_night,val_night,val2_select}   tile packs (see docs/ on pack building)
#   $NR_OUT/{runs,scores,gates,...}                                                     outputs of the runners
set -euo pipefail
: "${NR_DATA_ROOT:?set NR_DATA_ROOT to the data directory}"
NR_OUT="${NR_OUT:-$NR_DATA_ROOT/exp}"
PACKS="$NR_DATA_ROOT/packs"
PYTHON="${PYTHON:-python3}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$NR_OUT"
