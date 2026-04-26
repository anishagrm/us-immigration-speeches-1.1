#!/bin/bash
#SBATCH -J roosebert_mlm
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-gpu=16G
#SBATCH -t 12:00:00
#SBATCH --qos coc-ice
#SBATCH -o logs/roosebert_mlm_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agurram9@gatech.edu
#SBATCH --exclude=atl1-1-01-005-17-0

cd $SLURM_SUBMIT_DIR
mkdir -p logs

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate roosebert
module load cuda/12.9.1

MODEL=ddore14/RooseBERT-cont-cased
MODEL_TAG=RooseBERT-cont-cased
EMB_DIR=data/speeches/Congress/contextual-embeddings/${MODEL_TAG}
METAPHOR_DIR=data/speeches/Congress/metaphors/${MODEL_TAG}

echo "=== Caching model ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}', ignore_patterns=['trainer_state.json'])
"

# Stage 1: extract contextual embeddings with mention masking
echo "=== Stage 1: Extracting embeddings (${MODEL_TAG}) ==="
python3 -m embeddings.embed_immigrant_terms_masked \
    --model-type bert \
    --model ${MODEL} \
    --outdir ${EMB_DIR} \
    --device 0

# Stage 2: apply MLM head to get metaphor probabilities
echo "=== Stage 2: Converting embeddings to word probs (${MODEL_TAG}) ==="
python3 -m embeddings.convert_embeddings_to_word_probs \
    --model-type bert \
    --model ${MODEL} \
    --infile ${EMB_DIR}/immigrant_vectors_masked.npz \
    --outdir ${EMB_DIR} \
    --device 0

# Stage 3: run metaphorical analysis and statistical tests
echo "=== Stage 3: Metaphorical analysis (${MODEL_TAG}) ==="
python3 -m analysis.run_metaphorical_analysis \
    --emb-dir ${EMB_DIR} \
    --imm-groups-file data/speeches/Congress/tagged_counts/imm_mention_sent_indices_by_group.json \
    --outdir ${METAPHOR_DIR}

echo "=== Done. Embeddings in ${EMB_DIR}, results in ${METAPHOR_DIR} ==="
