#!/bin/bash
# Experiment D: guarded scoring by pairing-window bin.
# usage: NR_DATA_ROOT=<dir> scripts/exp/d_score.sh <git sha7> <runs dir> <ckpt file name> <with_all 0|1> <name> [<name> ...]
# SELECT tiles of the SELECT-only pack, bins |dt| in (0,5], (5,15], (15,30] (+ all SELECT when with_all=1), then the old 9-pair night validation overall (pack without tile ids: allowed with the documented reason,
# as for the stored old-validation scores). Output: $NR_OUT/scoresD/<name>/score_*.json
source "$(dirname "$0")/env.sh"
echo '{"const": -0.36296144127845764}' > "$NR_OUT/const.json"
sha=$1; rd=$2; ck=$3; wa=$4; shift 4
sc="$PYTHON scripts/score_run.py --train UNUSED --const-json $NR_OUT/const.json --runs-dir $rd --names $* --ckpt $ck --out $NR_OUT/scoresD --git-sha $sha --force"
for b in "0 5" "5 15" "15 30"; do $sc --data "$PACKS/val2_select" --set select --split configs/nf2_split_episodes.json --dt-range $b; done
[ "$wa" = 1 ] && $sc --data "$PACKS/val2_select" --set select --split configs/nf2_split_episodes.json
$sc --data "$PACKS/val_night" --set oldval --allow-unnamed-reason "old 9-pair night validation pack: ids not in nf2; same as the stored oldval scores"
