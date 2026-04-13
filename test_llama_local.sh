#!/bin/bash
# Smoke test for the Llama QLoRA pipeline on a local machine (MPS or CPU).
# Uses a 30-item subset of the tone data and runs 1 epoch of training + eval.
# NOTE: 4-bit quantization is automatically disabled on MPS/CPU (no bitsandbytes).
#
# Usage: conda run -n llama bash test_llama_local.sh

cd "$(dirname "$0")" || exit 1

PYTHON=$(conda run -n llama which python 2>/dev/null || python3 -c "import sys; print(sys.executable)")
echo "Using Python: $PYTHON"

# Create a 30-item smoke subset from the tone data
$PYTHON - <<'EOF'
import json, random, os
random.seed(0)
src = "data/speeches/Congress/tone/splits/label-weights/all.jsonlist"
dst_dir = "data/speeches/Congress/tone/splits/smoke_test_llama/"
os.makedirs(dst_dir, exist_ok=True)
with open(src) as f:
    items = [json.loads(l) for l in f if l.strip()]
subset = random.sample(items, 30)
# All 30 go into train; also create tiny dev/test splits
with open(os.path.join(dst_dir, "all.jsonlist"), "w") as f:
    for item in subset:
        f.write(json.dumps(item) + "\n")
with open(os.path.join(dst_dir, "dev.jsonlist"), "w") as f:
    for item in subset[:10]:
        f.write(json.dumps(item) + "\n")
with open(os.path.join(dst_dir, "test.jsonlist"), "w") as f:
    for item in subset[10:20]:
        f.write(json.dumps(item) + "\n")
print(f"Wrote {len(subset)} items to {dst_dir}")
EOF

echo "=== Starting 1-epoch Llama smoke test (Llama-3.2-1B, local only) ==="
export TRANSFORMERS_VERBOSITY=info
export HF_HUB_VERBOSITY=info
$PYTHON -m classification.run_llama_qlora \
    --model meta-llama/Llama-3.2-1B \
    --basedir data/speeches/Congress/tone/splits/smoke_test_llama/ \
    --train-file all.jsonlist \
    --dev-file dev.jsonlist \
    --test-file test.jsonlist \
    --output-prefix smoke \
    --do-train \
    --do-eval \
    --n-epochs 1 \
    --batch-size 2 \
    --grad-accum 2 \
    --max-seq-length 256 \
    --eval-batch-size 4 \
    --seed 42 \
    --no-4bit

echo "=== Smoke test complete. Check for 'Device: mps' or 'Device: cpu' above ==="
