#!/bin/bash
#SBATCH -J llama_qlora_tone
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=40G
#SBATCH -t 12:00:00
#SBATCH -o logs/llama_qlora_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agurram9@gatech.edu

# Llama-3.1-8B QLoRA tone classifier
# Requires: conda env 'llama' with transformers, peft, bitsandbytes, trl, tqdm

cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
mkdir -p logs

module load anaconda3
source activate llama

echo "=== Caching Llama-3.1-8B ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B')
print('Model cached.')
"

echo "=== Training tone classifier (QLoRA) ==="
srun python -m classification.run_llama_qlora \
    --model meta-llama/Llama-3.1-8B \
    --basedir data/speeches/Congress/tone/splits/label-weights/ \
    --train-file all.jsonlist \
    --dev-file dev.jsonlist \
    --test-file test.jsonlist \
    --output-prefix llama_qlora \
    --do-train \
    --do-eval \
    --n-epochs 3 \
    --lr 2e-4 \
    --batch-size 4 \
    --grad-accum 4 \
    --max-seq-length 512 \
    --lora-rank 16 \
    --lora-alpha 16 \
    --eval-batch-size 8 \
    --seed 42

echo "=== Done ==="
