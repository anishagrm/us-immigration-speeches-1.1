#!/bin/bash
# Quick sanity check: trains on 100 examples for 1 epoch using MPS (M1 GPU).
# Run this locally to verify the MPS patch and RooseBERT loading work before
# submitting the full job to PACE ICE.

cd "$(dirname "$0")" || exit 1

# Create a tiny 100-item subset for the smoke test
python3 - <<'EOF'
import json, random, os
random.seed(0)
src = "data/speeches/Congress/relevance/splits/basic/all.jsonlist"
dst_dir = "data/speeches/Congress/relevance/splits/smoke_test/"
os.makedirs(dst_dir, exist_ok=True)
with open(src) as f:
    items = [json.loads(l) for l in f if l.strip()]
subset = random.sample(items, 100)
with open(os.path.join(dst_dir, "all.jsonlist"), "w") as f:
    for item in subset:
        f.write(json.dumps(item) + "\n")
print(f"Wrote {len(subset)} items to {dst_dir}all.jsonlist")
EOF

echo "=== Starting 1-epoch smoke test (MPS device expected) ==="
python3 -m classification.run_final_model \
    --model_type bert \
    --model_name_or_path ddore14/RooseBERT-cont-uncased \
    --basedir data/speeches/Congress/relevance/splits/smoke_test/ \
    --train-file all.jsonlist \
    --output-prefix smoke \
    --n_epochs 1 \
    --lr 2e-5 \
    --max_seq_length 128 \
    --per_gpu 2 \
    --seed 42

echo "=== Smoke test complete. Check logs above for 'device: mps' ==="
