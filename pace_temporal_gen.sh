#!/bin/bash
#SBATCH -J temporal_gen
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --mem-per-gpu=16G
#SBATCH -t 24:00:00
#SBATCH -o logs/temporal_gen_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jvarma3@gatech.edu

cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
mkdir -p logs

module load anaconda3
conda activate hum

echo "=== Experiment 3: Temporal Generalization ==="
srun python -m classification.run_temporal_generalization \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --split basic \
    --n_epochs 7 \
    --lr 2e-5 \
    --per_gpu 4 \
    --max_seq_length 512 \
    --seed 42 \
    --results-dir results/temporal_generalization

echo "=== Done ==="
