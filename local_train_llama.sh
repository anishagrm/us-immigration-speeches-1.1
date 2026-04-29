#!/bin/bash
# Local fine-tuning on MPS (Apple Silicon).
# 2 epochs, 50% of training data, max_seq_length=256.
# Expected: ~20 min tone + ~40 min relevance.
# Usage: bash local_train_llama.sh

set -euo pipefail

PYTHON=$(which python)
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== Local Llama QLoRA Fine-tuning ==="
date

$PYTHON -c "
import random
random.seed(42)
for src, dst in [
    ('data/speeches/Congress/tone/splits/label-weights/all.jsonlist', '$TMPDIR/tone_train.jsonlist'),
    ('data/speeches/Congress/relevance/splits/basic/all.jsonlist', '$TMPDIR/rel_train.jsonlist'),
]:
    lines = [l for l in open(src) if l.strip()]
    sample = random.sample(lines, len(lines) // 2)
    open(dst, 'w').writelines(sample)
    print(f'{src}: using {len(sample)}/{len(lines)} examples')
"

echo ""
echo "--- [1/2] tone ---"
$PYTHON -m classification.run_llama_qlora \
    --model meta-llama/Llama-3.2-1B \
    --basedir data/speeches/Congress/tone/splits/label-weights/ \
    --train-file $TMPDIR/tone_train.jsonlist \
    --dev-file dev.jsonlist \
    --test-file test.jsonlist \
    --outdir llama-runs/run2/tone \
    --do-train \
    --do-eval \
    --n-epochs 2 \
    --lr 2e-4 \
    --batch-size 4 \
    --grad-accum 2 \
    --max-seq-length 256 \
    --lora-rank 16 \
    --lora-alpha 16 \
    --logging-steps 20 \
    --eval-batch-size 8 \
    --seed 42

echo ""
echo "--- [2/2] relevance ---"
$PYTHON -m classification.run_llama_qlora_relevance \
    --model meta-llama/Llama-3.2-1B \
    --basedir data/speeches/Congress/relevance/splits/basic/ \
    --train-file $TMPDIR/rel_train.jsonlist \
    --dev-file dev.jsonlist \
    --test-file test.jsonlist \
    --outdir llama-runs/run2/relevance \
    --do-train \
    --do-eval \
    --n-epochs 2 \
    --lr 2e-4 \
    --batch-size 4 \
    --grad-accum 2 \
    --max-seq-length 256 \
    --lora-rank 16 \
    --lora-alpha 16 \
    --logging-steps 20 \
    --eval-batch-size 8 \
    --seed 42

echo ""
echo "=== Done ==="
date
