#!/bin/bash
#SBATCH -J model_compare
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --mem-per-gpu=16G
#SBATCH -t 10:00:00
#SBATCH -o logs/compare_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agurram9@gatech.edu

cd $SLURM_SUBMIT_DIR
mkdir -p logs

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate roosebert
module load cuda/12.9.1

echo "=== Caching models ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('roberta-base', ignore_patterns=['trainer_state.json'])
snapshot_download('ddore14/RooseBERT-cont-uncased', ignore_patterns=['trainer_state.json'])
"

RELEVANCE_DIR=data/speeches/Congress/relevance/splits/basic
TONE_DIR=data/speeches/Congress/tone/splits/label-weights

# --- Relevance: RoBERTa ---
echo "=== Relevance: RoBERTa ==="
python3 -m hf.run \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --name compare \
    --do_train --train train.jsonlist \
    --do_eval --eval_partition test \
    --data_dir $RELEVANCE_DIR \
    --output_dir $RELEVANCE_DIR/compare/roberta-base \
    --max_seq_length 400 \
    --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
    --learning_rate 2e-5 --num_train_epochs 7 \
    --overwrite_cache --overwrite_output_dir \
    --weight_field weight --metrics accuracy,f1 \
    --seed 42 --save_steps 0

# --- Relevance: RooseBERT ---
echo "=== Relevance: RooseBERT ==="
python3 -m hf.run \
    --model_type bert \
    --model_name_or_path ddore14/RooseBERT-cont-uncased \
    --name compare \
    --do_train --train train.jsonlist \
    --do_eval --eval_partition test \
    --data_dir $RELEVANCE_DIR \
    --output_dir $RELEVANCE_DIR/compare/roosebert \
    --max_seq_length 512 \
    --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
    --learning_rate 2e-5 --num_train_epochs 7 \
    --overwrite_cache --overwrite_output_dir \
    --weight_field weight --metrics accuracy,f1 \
    --seed 42 --save_steps 0 --do_lower_case

# --- Tone: RoBERTa ---
echo "=== Tone: RoBERTa ==="
python3 -m hf.run \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --name compare \
    --do_train --train train.jsonlist \
    --do_eval --eval_partition test \
    --data_dir $TONE_DIR \
    --output_dir $TONE_DIR/compare/roberta-base \
    --max_seq_length 400 \
    --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
    --learning_rate 2e-5 --num_train_epochs 7 \
    --overwrite_cache --overwrite_output_dir \
    --metrics accuracy,per_class_f1 \
    --seed 42 --save_steps 0

# --- Tone: RooseBERT ---
echo "=== Tone: RooseBERT ==="
python3 -m hf.run \
    --model_type bert \
    --model_name_or_path ddore14/RooseBERT-cont-uncased \
    --name compare \
    --do_train --train train.jsonlist \
    --do_eval --eval_partition test \
    --data_dir $TONE_DIR \
    --output_dir $TONE_DIR/compare/roosebert \
    --max_seq_length 512 \
    --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
    --learning_rate 2e-5 --num_train_epochs 7 \
    --overwrite_cache --overwrite_output_dir \
    --metrics accuracy,per_class_f1 \
    --seed 42 --save_steps 0 --do_lower_case

echo "=== Done ==="
