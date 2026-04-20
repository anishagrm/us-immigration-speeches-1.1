#!/bin/bash
#SBATCH -J temporal_gen_cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH -o logs/temporal_gen_cpu_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jvarma3@gatech.edu

cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
mkdir -p logs

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hum

export HF_HOME=$HOME/scratch/hf_cache
mkdir -p $HF_HOME

echo "=== Pre-caching roberta-base ==="
python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('roberta-base'); AutoModel.from_pretrained('roberta-base'); print('Cached.')"

echo "=== Experiment 3: Temporal Generalization (CPU) ==="
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
    --overwrite

echo "=== Done ==="
