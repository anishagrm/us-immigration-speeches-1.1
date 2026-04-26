#!/bin/bash
#SBATCH -J final_comparison
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --mem-per-gpu=16G
#SBATCH -t 16:00:00
#SBATCH -o logs/final_comparison_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agurram9@gatech.edu

cd $SLURM_SUBMIT_DIR
mkdir -p logs runs/experiment_3

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate roosebert
module load cuda/12.9.1

RELEVANCE_DIR=data/speeches/Congress/relevance/splits/basic
TONE_DIR=data/speeches/Congress/tone/splits/label-weights
RESULTS_DIR=runs/experiment_3

cleanup() {
    local dir=$1
    local label=$2
    cp $dir/eval_results_test.txt $RESULTS_DIR/${label}.txt 2>/dev/null || echo "Warning: no eval_results_test.txt in $dir"
    rm -f $dir/pytorch_model.bin $dir/optimizer.pt $dir/scheduler.pt
    rm -rf $dir/checkpoint-*
    echo "Saved $RESULTS_DIR/${label}.txt"
}

for SEED in 0 1 2; do
    echo "=========================================="
    echo "SEED = $SEED"
    echo "=========================================="

    # --- Relevance ---
    # Best: cont-cased @ lr=3e-5

    echo "--- Relevance: RoBERTa (seed=$SEED) ---"
    python3 -m hf.run \
        --model_type roberta \
        --model_name_or_path roberta-base \
        --name compare \
        --do_train --train train.jsonlist \
        --do_eval --eval_partition test \
        --data_dir $RELEVANCE_DIR \
        --output_dir $RELEVANCE_DIR/final/roberta_s${SEED} \
        --max_seq_length 400 \
        --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
        --learning_rate 2e-5 --num_train_epochs 7 \
        --overwrite_cache --overwrite_output_dir \
        --weight_field weight --metrics accuracy,f1 \
        --seed $SEED --save_steps 0
    cleanup $RELEVANCE_DIR/final/roberta_s${SEED} relevance_roberta_s${SEED}

    echo "--- Relevance: RooseBERT-cont-cased lr=3e-5 (seed=$SEED) ---"
    python3 -m hf.run \
        --model_type bert \
        --model_name_or_path ddore14/RooseBERT-cont-cased \
        --name compare \
        --do_train --train train.jsonlist \
        --do_eval --eval_partition test \
        --data_dir $RELEVANCE_DIR \
        --output_dir $RELEVANCE_DIR/final/cont-cased_lr3e-5_s${SEED} \
        --max_seq_length 512 \
        --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
        --learning_rate 3e-5 --num_train_epochs 7 \
        --overwrite_cache --overwrite_output_dir \
        --weight_field weight --metrics accuracy,f1 \
        --seed $SEED --save_steps 0
    cleanup $RELEVANCE_DIR/final/cont-cased_lr3e-5_s${SEED} relevance_cont-cased_lr3e-5_s${SEED}

    # --- Tone ---
    # Best: scr-cased @ lr=1e-5

    echo "--- Tone: RoBERTa (seed=$SEED) ---"
    python3 -m hf.run \
        --model_type roberta \
        --model_name_or_path roberta-base \
        --name compare \
        --do_train --train train.jsonlist \
        --do_eval --eval_partition test \
        --data_dir $TONE_DIR \
        --output_dir $TONE_DIR/final/roberta_s${SEED} \
        --max_seq_length 400 \
        --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
        --learning_rate 2e-5 --num_train_epochs 7 \
        --overwrite_cache --overwrite_output_dir \
        --metrics accuracy,per_class_f1 \
        --seed $SEED --save_steps 0
    cleanup $TONE_DIR/final/roberta_s${SEED} tone_roberta_s${SEED}

    echo "--- Tone: RooseBERT-scr-cased lr=1e-5 (seed=$SEED) ---"
    python3 -m hf.run \
        --model_type bert \
        --model_name_or_path ddore14/RooseBERT-scr-cased \
        --name compare \
        --do_train --train train.jsonlist \
        --do_eval --eval_partition test \
        --data_dir $TONE_DIR \
        --output_dir $TONE_DIR/final/scr-cased_lr1e-5_s${SEED} \
        --max_seq_length 512 \
        --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
        --learning_rate 1e-5 --num_train_epochs 7 \
        --overwrite_cache --overwrite_output_dir \
        --metrics accuracy,per_class_f1 \
        --seed $SEED --save_steps 0
    cleanup $TONE_DIR/final/scr-cased_lr1e-5_s${SEED} tone_scr-cased_lr1e-5_s${SEED}

done

echo "=== Final comparison done. Results in $RESULTS_DIR ==="
