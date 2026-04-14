#!/bin/bash
#SBATCH -J roberta_folds
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --mem-per-gpu=16G
#SBATCH -t 16:00:00
#SBATCH -o logs/roberta_folds_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agurram9@gatech.edu

cd $SLURM_SUBMIT_DIR
mkdir -p logs

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate roosebert
module load cuda/12.9.1

echo "=== Caching roberta-base ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('roberta-base', ignore_patterns=['trainer_state.json'])
"

echo "=== Relevance folds: RoBERTa ==="
python3 -m classification.run_folds_hf \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --basedir data/speeches/Congress/relevance/splits/basic/ \
    --train-file train.jsonlist \
    --output-prefix roberta-base \
    --n_epochs 7 \
    --lr 2e-5 \
    --max_seq_length 400 \
    --per_gpu 8 \
    --seed 42

echo "=== Tone folds: RoBERTa ==="
python3 -m classification.run_folds_hf_tone \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --basedir data/speeches/Congress/tone/splits/label-weights/ \
    --train-file train.jsonlist \
    --output-prefix roberta-base \
    --n_epochs 7 \
    --lr 2e-5 \
    --max_seq_length 400 \
    --per_gpu 8 \
    --seed 42

echo "=== Done ==="
