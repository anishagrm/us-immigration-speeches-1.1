#!/bin/bash
#SBATCH -J roosebert_imm                              
#SBATCH -N 1
#SBATCH --ntasks-per-node=4                                                                                     
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16G                                                                                       
#SBATCH -t 6:00:00                                    
#SBATCH -o logs/roosebert_%j.out                                                                                
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jvarma3@gatech.edu

cd $SLURM_SUBMIT_DIR
mkdir -p logs

module load anaconda3
conda activate roosebert

echo "=== Caching model ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ddore14/RooseBERT-cont-uncased', ignore_patterns=['trainer_state.json'])
"

echo "=== Training relevance classifier ==="
srun python3 -m classification.run_final_model \
    --model_type bert \
    --model_name_or_path ddore14/RooseBERT-cont-uncased \
    --basedir data/speeches/Congress/relevance/splits/basic/ \
    --train-file all.jsonlist \
    --output-prefix roosebert \
    --n_epochs 7 \
    --lr 2e-5 \
    --max_seq_length 512 \
    --per_gpu 8 \
    --seed 42

echo "=== Training tone classifier ==="
srun python3 -m classification.run_final_model_tone \
    --model_type bert \
    --model_name_or_path ddore14/RooseBERT-cont-uncased \
    --basedir data/speeches/Congress/tone/splits/label-weights/ \
    --train-file all.jsonlist \
    --output-prefix roosebert \
    --n_epochs 7 \
    --lr 2e-5 \
    --max_seq_length 512 \
    --per_gpu 8 \
    --seed 42
echo "=== Done ==="