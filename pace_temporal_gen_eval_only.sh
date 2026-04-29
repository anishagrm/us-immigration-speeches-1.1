#!/bin/bash
#SBATCH -J temporal_gen_eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/temporal_gen_eval_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jvarma3@gatech.edu

cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
mkdir -p logs

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llama

export HF_HOME=$HOME/scratch/hf_cache
mkdir -p $HF_HOME

echo "=== Verifying checkpoints ==="
for era in early mid modern; do
    ckpt="data/annotations/relevance_and_tone/${era}/relevance/splits/basic/bert/temporal_${era}_roberta-base_s42_lr2e-05_msl512"
    if ls "$ckpt"/model.safetensors "$ckpt"/pytorch_model.bin 2>/dev/null | head -1 | grep -q .; then
        echo "  [OK] $era checkpoint found"
    else
        echo "  [MISSING] $era checkpoint: $ckpt"
        echo "  Contents: $(ls $ckpt 2>/dev/null || echo 'directory does not exist')"
    fi
done

echo "=== Experiment 3: Temporal Generalization (eval only) ==="
srun python -m classification.run_temporal_generalization \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --split basic \
    --n_epochs 7 \
    --lr 2e-5 \
    --per_gpu 8 \
    --max_seq_length 512 \
    --seed 42 \
    --results-dir results/temporal_generalization \
    --eval-only

echo "=== Done ==="
