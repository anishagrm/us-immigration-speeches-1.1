#!/bin/bash
# Local fine-tuning for both Llama QLoRA tasks on MPS (Apple Silicon).
# Estimated runtime: ~60 min tone + ~2 hr relevance = ~3 hr total.
#
# Differences from PACE run:
#   - max_seq_length=256 instead of 512 (faster on MPS)
#   - fp32 (4-bit quantization not supported on MPS)
#   - output-prefix: llama_qlora_local (separate from PACE results)
#
# Usage (from repo root, with llama conda env active):
#   bash local_train_llama.sh

set -euo pipefail

PYTHON=$(which python)

echo "=== Local Llama QLoRA Fine-tuning ==="
echo "Device: MPS (Apple Silicon)"
echo "Estimated time: ~60 min total"
echo ""

# ── Tone classifier ───────────────────────────────────────────────────────────
echo "--- [1/2] Tone classifier (anti / neutral / pro) ---"
echo "Expected: ~60 min"
date

$PYTHON -m classification.run_llama_qlora \
    --model meta-llama/Llama-3.2-1B \
    --basedir data/speeches/Congress/tone/splits/label-weights/ \
    --train-file all.jsonlist \
    --dev-file dev.jsonlist \
    --test-file test.jsonlist \
    --outdir llama-runs/run2/tone \
    --do-train \
    --do-eval \
    --n-epochs 3 \
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

# ── Relevance classifier ──────────────────────────────────────────────────────
echo "--- [2/2] Relevance classifier (yes / no) ---"
echo "Expected: ~2 hr"
date

$PYTHON -m classification.run_llama_qlora_relevance \
    --model meta-llama/Llama-3.2-1B \
    --basedir data/speeches/Congress/relevance/splits/basic/ \
    --train-file all.jsonlist \
    --dev-file dev.jsonlist \
    --test-file test.jsonlist \
    --outdir llama-runs/run2/relevance \
    --do-train \
    --do-eval \
    --n-epochs 3 \
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
