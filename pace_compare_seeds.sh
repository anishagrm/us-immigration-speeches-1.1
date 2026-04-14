#!/bin/bash
#SBATCH -J compare_seeds
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --mem-per-gpu=16G
#SBATCH -t 16:00:00
#SBATCH -o logs/compare_seeds_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agurram9@gatech.edu

cd $SLURM_SUBMIT_DIR
mkdir -p logs

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate roosebert
module load cuda/12.9.1

RELEVANCE_DIR=data/speeches/Congress/relevance/splits/basic
TONE_DIR=data/speeches/Congress/tone/splits/label-weights

cleanup() {
    local dir=$1
    rm -f $dir/pytorch_model.bin
    rm -f $dir/optimizer.pt
    rm -f $dir/scheduler.pt
    rm -rf $dir/checkpoint-*
    echo "Cleaned up model weights in $dir"
}

for SEED in 0 1 2; do
    echo "=========================================="
    echo "SEED = $SEED"
    echo "=========================================="

    echo "--- Relevance: RoBERTa (seed=$SEED) ---"
    python3 -m hf.run \
        --model_type roberta \
        --model_name_or_path roberta-base \
        --name compare \
        --do_train --train train.jsonlist \
        --do_eval --eval_partition test \
        --data_dir $RELEVANCE_DIR \
        --output_dir $RELEVANCE_DIR/compare/seeds/roberta_s${SEED} \
        --max_seq_length 400 \
        --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
        --learning_rate 2e-5 --num_train_epochs 7 \
        --overwrite_cache --overwrite_output_dir \
        --weight_field weight --metrics accuracy,f1 \
        --seed $SEED --save_steps 0
    cleanup $RELEVANCE_DIR/compare/seeds/roberta_s${SEED}

    echo "--- Relevance: RooseBERT (seed=$SEED) ---"
    python3 -m hf.run \
        --model_type bert \
        --model_name_or_path ddore14/RooseBERT-cont-uncased \
        --name compare \
        --do_train --train train.jsonlist \
        --do_eval --eval_partition test \
        --data_dir $RELEVANCE_DIR \
        --output_dir $RELEVANCE_DIR/compare/seeds/roosebert_s${SEED} \
        --max_seq_length 512 \
        --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
        --learning_rate 2e-5 --num_train_epochs 7 \
        --overwrite_cache --overwrite_output_dir \
        --weight_field weight --metrics accuracy,f1 \
        --seed $SEED --save_steps 0 --do_lower_case
    cleanup $RELEVANCE_DIR/compare/seeds/roosebert_s${SEED}

    echo "--- Tone: RoBERTa (seed=$SEED) ---"
    python3 -m hf.run \
        --model_type roberta \
        --model_name_or_path roberta-base \
        --name compare \
        --do_train --train train.jsonlist \
        --do_eval --eval_partition test \
        --data_dir $TONE_DIR \
        --output_dir $TONE_DIR/compare/seeds/roberta_s${SEED} \
        --max_seq_length 400 \
        --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
        --learning_rate 2e-5 --num_train_epochs 7 \
        --overwrite_cache --overwrite_output_dir \
        --metrics accuracy,per_class_f1 \
        --seed $SEED --save_steps 0
    cleanup $TONE_DIR/compare/seeds/roberta_s${SEED}

    echo "--- Tone: RooseBERT (seed=$SEED) ---"
    python3 -m hf.run \
        --model_type bert \
        --model_name_or_path ddore14/RooseBERT-cont-uncased \
        --name compare \
        --do_train --train train.jsonlist \
        --do_eval --eval_partition test \
        --data_dir $TONE_DIR \
        --output_dir $TONE_DIR/compare/seeds/roosebert_s${SEED} \
        --max_seq_length 512 \
        --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
        --learning_rate 2e-5 --num_train_epochs 7 \
        --overwrite_cache --overwrite_output_dir \
        --metrics accuracy,per_class_f1 \
        --seed $SEED --save_steps 0 --do_lower_case
    cleanup $TONE_DIR/compare/seeds/roosebert_s${SEED}

done

echo "=== All seeds done ==="
