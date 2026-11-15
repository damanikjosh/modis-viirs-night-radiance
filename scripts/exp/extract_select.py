#!/usr/bin/env python
"""Paper experiments: extract ONLY the SELECT tiles (by id, configs/nf2_split.json) from the night-fire wave-2 val2 shard tars. REPORT/TRAIN members are never read (the tar is
walked by header; data of other members is skipped). Prints counts. Usage: extract_select.py <out dir> <tar> [<tar> ...]"""
import json, os, sys, tarfile
split = json.load(open("configs/nf2_split.json"))["tiles"]; sel = set(split["select"]); other = set(split["report"]) | set(split["train"])
out = sys.argv[1]; os.makedirs(out, exist_ok=True); got = skipped = 0
for t in sys.argv[2:]:
    with tarfile.open(t) as tf:
        for m in tf:
            n = os.path.basename(m.name)[:-4] if m.name.endswith(".npz") else None
            if n in sel:
                with tf.extractfile(m) as src, open(os.path.join(out, n + ".npz"), "wb") as dst:
                    dst.write(src.read())
                got += 1
            else:
                skipped += 1
print(json.dumps({"extracted_select": got, "skipped_other_members": skipped, "select_list": len(sel)}))
assert not (set(f[:-4] for f in os.listdir(out)) & other), "a non-SELECT tile was extracted"
