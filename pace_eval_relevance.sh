#!/bin/bash
#SBATCH -J relevance_eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --mem-per-gpu=16G
#SBATCH -t 1:00:00
#SBATCH -o logs/relevance_eval_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agurram9@gatech.edu

cd $SLURM_SUBMIT_DIR
mkdir -p logs

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate roosebert
module load cuda/12.9.1

RELEVANCE_DIR=data/speeches/Congress/relevance/splits/basic

echo "=== Relevance eval: RoBERTa ==="
python3 -m hf.run \
    --model_type roberta \
    --model_name_or_path $RELEVANCE_DIR/compare/roberta-base \
    --name compare \
    --do_eval --eval_partition test \
    --data_dir $RELEVANCE_DIR \
    --output_dir $RELEVANCE_DIR/compare/roberta-base \
    --max_seq_length 400 \
    --per_gpu_eval_batch_size 8 \
    --overwrite_cache \
    --weight_field weight --metrics accuracy,f1 \
    --seed 42

echo "=== Relevance eval: RooseBERT ==="
python3 -m hf.run \
    --model_type bert \
    --model_name_or_path $RELEVANCE_DIR/compare/roosebert \
    --name compare \
    --do_eval --eval_partition test \
    --data_dir $RELEVANCE_DIR \
    --output_dir $RELEVANCE_DIR/compare/roosebert \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size 8 \
    --overwrite_cache \
    --weight_field weight --metrics accuracy,f1 \
    --seed 42 --do_lower_case

echo "=== Done ==="
