#!/bin/bash
# Local smoke test: verifies Llama QLoRA training loss is non-zero before
# submitting to PACE. Runs ~10 steps on a 40-example sample, MPS/CPU only.
#
# Usage (from repo root):
#   bash smoke_test_llama_local.sh
#
# Exit 0 = PASS (safe to submit), Exit 1 = FAIL (do not submit)

set -euo pipefail

PYTHON=${PYTHON:-$(which python)}
N_SAMPLES=40
MAX_SEQ=256
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== Llama QLoRA Local Smoke Test ==="
echo "Samples: $N_SAMPLES | Max seq len: $MAX_SEQ"
echo ""

PASS=true

run_test() {
    local task=$1
    local script=$2
    local train_src=$3

    echo "--- $task ---"

    if [ ! -f "$train_src" ]; then
        echo "SKIP: training data not found at $train_src"
        return
    fi

    # Sample N random lines from the training data
    local tmpbase="$TMPDIR/$task"
    mkdir -p "$tmpbase"
    $PYTHON -c "
import json, random, sys
lines = [l for l in open('$train_src') if l.strip()]
random.seed(42)
for l in random.sample(lines, min($N_SAMPLES, len(lines))):
    sys.stdout.write(l)
" > "$tmpbase/all.jsonlist"

    local log="$TMPDIR/${task}.log"

    # Run training (4-bit auto-disabled on MPS/CPU)
    $PYTHON -m "$script" \
        --model meta-llama/Llama-3.2-1B \
        --basedir "$tmpbase/" \
        --train-file all.jsonlist \
        --output-prefix smoke \
        --do-train \
        --n-epochs 1 \
        --lr 2e-4 \
        --batch-size 4 \
        --grad-accum 1 \
        --max-seq-length $MAX_SEQ \
        --lora-rank 16 \
        --lora-alpha 16 \
        --logging-steps 1 \
        --seed 42 \
        2>&1 | tee "$log"

    # Parse all logged loss values and the final train_loss
    local result
    result=$($PYTHON -c "
import re, sys

losses = []
for line in open('$log'):
    # per-step: {'loss': '0.1234', ...}
    for m in re.finditer(r\"'loss':\\s*'([0-9.e+-]+)'\", line):
        losses.append(float(m.group(1)))
    # final summary: 'train_loss': '0.1234'
    m = re.search(r\"'train_loss':\\s*'([0-9.e+-]+)'\", line)
    if m:
        losses.append(float(m.group(1)))

if not losses:
    print('NO_LOSS_LOGGED')
elif all(v == 0.0 for v in losses):
    print('ALL_ZERO')
else:
    nonzero = [v for v in losses if v != 0.0]
    suffix = '...' if len(losses) > 5 else ''
    print(f'OK  losses={losses[:5]}{suffix}  nonzero={len(nonzero)}/{len(losses)}')
")

    if [[ "$result" == OK* ]]; then
        echo "PASS: $task — $result"
    else
        echo "FAIL: $task — $result"
        PASS=false
    fi
    echo ""
}

run_test "tone" \
    "classification.run_llama_qlora" \
    "data/speeches/Congress/tone/splits/label-weights/all.jsonlist"

run_test "relevance" \
    "classification.run_llama_qlora_relevance" \
    "data/speeches/Congress/relevance/splits/basic/all.jsonlist"

if [ "$PASS" = true ]; then
    echo "=== PASSED — safe to submit to PACE ==="
    exit 0
else
    echo "=== FAILED — fix the bug before submitting ==="
    exit 1
fi
