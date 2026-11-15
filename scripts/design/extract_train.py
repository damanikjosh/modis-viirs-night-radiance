#!/usr/bin/env python
"""extract ONLY the TRAIN tiles (by id, configs/nf2_split.json) from the night-fire wave-2 train shard tars. Members of other tiles (REPORT included) are skipped by header and their data never read.
Usage: extract_train.py <out dir> <tar> [<tar> ...]"""
import json, os, sys, tarfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard
allowed = guard.allowed_ids(sides=("train",)); out = sys.argv[1]; os.makedirs(out, exist_ok=True); got = skipped = 0
for t in sys.argv[2:]:
    with tarfile.open(t) as tf:
        for m in tf:
            n = os.path.basename(m.name)[:-4] if m.name.endswith(".npz") else None
            if n in allowed:
                with tf.extractfile(m) as src, open(os.path.join(out, n + ".npz"), "wb") as dst:
                    dst.write(src.read())
                got += 1
            else:
                skipped += 1
rep = {"extracted_train": got, "skipped_other_members": skipped, "train_list": len(allowed)}
print(json.dumps(rep)); json.dump(rep, open(os.path.join(out, "_extract_report.json"), "w"))
guard.check_ids([f[:-4] for f in os.listdir(out) if f.endswith(".npz")], allowed)
