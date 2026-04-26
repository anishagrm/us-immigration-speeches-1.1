#!/bin/bash
#SBATCH -J lr_sweep
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-v100
#SBATCH --mem-per-gpu=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/lr_sweep_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agurram9@gatech.edu

cd $SLURM_SUBMIT_DIR
mkdir -p logs runs/experiment_2

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate roosebert
module load cuda/12.9.1

RELEVANCE_DIR=data/speeches/Congress/relevance/splits/basic
TONE_DIR=data/speeches/Congress/tone/splits/label-weights
RESULTS_DIR=runs/experiment_2

echo "=== Caching models ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ddore14/RooseBERT-cont-cased', ignore_patterns=['trainer_state.json'])
snapshot_download('ddore14/RooseBERT-scr-cased', ignore_patterns=['trainer_state.json'])
"

cleanup() {
    local dir=$1
    local label=$2
    cp $dir/eval_results_dev.txt $RESULTS_DIR/${label}.txt 2>/dev/null || echo "Warning: no eval_results_dev.txt in $dir"
    rm -f $dir/pytorch_model.bin $dir/optimizer.pt $dir/scheduler.pt
    rm -rf $dir/checkpoint-*
    echo "Saved $RESULTS_DIR/${label}.txt"
}

for MODEL in cont-cased scr-cased; do
    for LR in 1e-5 2e-5 3e-5; do

        echo "====== $MODEL | lr=$LR ======"

        echo "--- Relevance: RooseBERT-$MODEL lr=$LR ---"
        python3 -m hf.run \
            --model_type bert \
            --model_name_or_path ddore14/RooseBERT-${MODEL} \
            --name lrsweep \
            --do_train --train train.jsonlist \
            --do_eval --eval_partition dev \
            --data_dir $RELEVANCE_DIR \
            --output_dir $RELEVANCE_DIR/lrsweep/${MODEL}_lr${LR} \
            --max_seq_length 512 \
            --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
            --learning_rate $LR --num_train_epochs 7 \
            --overwrite_cache --overwrite_output_dir \
            --weight_field weight --metrics accuracy,f1 \
            --seed 42 --save_steps 0
        cleanup $RELEVANCE_DIR/lrsweep/${MODEL}_lr${LR} relevance_${MODEL}_lr${LR}

        echo "--- Tone: RooseBERT-$MODEL lr=$LR ---"
        python3 -m hf.run \
            --model_type bert \
            --model_name_or_path ddore14/RooseBERT-${MODEL} \
            --name lrsweep \
            --do_train --train train.jsonlist \
            --do_eval --eval_partition dev \
            --data_dir $TONE_DIR \
            --output_dir $TONE_DIR/lrsweep/${MODEL}_lr${LR} \
            --max_seq_length 512 \
            --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 \
            --learning_rate $LR --num_train_epochs 7 \
            --overwrite_cache --overwrite_output_dir \
            --metrics accuracy,per_class_f1 \
            --seed 42 --save_steps 0
        cleanup $TONE_DIR/lrsweep/${MODEL}_lr${LR} tone_${MODEL}_lr${LR}

    done
done

echo "=== LR sweep done. Results in $RESULTS_DIR ==="
