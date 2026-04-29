#!/bin/bash
#SBATCH -J llama_qlora_relevance
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --mem-per-gpu=16G
#SBATCH -t 12:00:00
#SBATCH -o llama-runs/run2/llama_qlora_relevance_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jvarma3@gatech.edu

# run2: fixes truncation bug where label token was cut off for long speeches,
# causing loss=0 throughout training. Speech text is now truncated to a budget
# that guarantees the label token survives. ZeroLossCallback aborts early if
# loss stays 0 after 5 logging steps.

cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

module load anaconda3
module load cuda/12.9.1
conda activate llama

export HF_HOME=$HOME/scratch/hf_cache
mkdir -p $HF_HOME
if [ -f $HOME/.cache/huggingface/token ] && [ ! -f $HF_HOME/token ]; then
    cp $HOME/.cache/huggingface/token $HF_HOME/token
fi

echo "=== Caching Llama-3.2-1B ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.2-1B')
print('Model cached.')
"

echo "=== Training relevance classifier (QLoRA, run2) ==="
srun python3 -m classification.run_llama_qlora_relevance \
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
    --grad-accum 4 \
    --max-seq-length 512 \
    --lora-rank 16 \
    --lora-alpha 16 \
    --eval-batch-size 8 \
    --seed 42

echo "=== Done ==="
