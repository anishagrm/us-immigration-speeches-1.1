
# Environment set up:
1. Log in to PACE: ssh [gt-username]@login-ice.pace.gatech.edu
2. upload the repo to pace and configure vscode to connect to host so u can develop on pace directly
3. Copy the `data/` dir to your scratch dir on PACE, then symlink it into the repo. If `data/` already exists in the repo, remove it first so the symlink replaces it (not nest inside it):
```bash
# Remove existing empty data dir in the repo (if present)
rm -rf ~/personal/us-immigration-speeches-1.1/data

# Symlink scratch data into the repo
ln -s ~/scratch/data ~/personal/us-immigration-speeches-1.1/data
```
4. Make a conda environment and move it to scratch (home dir has a small quota):
```bash
module load anaconda3
conda create -n llama python=3.11 -y

# PACE home dir has a small quota — move the env to scratch and symlink it
mkdir -p $HOME/scratch/envs
mv $HOME/.conda/envs/llama $HOME/scratch/envs/llama
ln -s $HOME/scratch/envs/llama $HOME/.conda/envs/llama

# Install packages using the full path to avoid conda activate issues
$HOME/scratch/envs/llama/bin/pip install "transformers>=4.40" "peft>=0.10" pandas numpy tqdm scikit-learn accelerate bitsandbytes huggingface_hub

# Install PyTorch with CUDA support (PACE uses CUDA 12.9, cu124 is compatible)
$HOME/scratch/envs/llama/bin/pip uninstall torch -y
$HOME/scratch/envs/llama/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124

# Verify GPU is visible
$HOME/scratch/envs/llama/bin/python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
# Should print: 2.x.x+cu124 and 12.4
```
Note: scratch storage persists across jobs but is wiped at semester end — back up results externally.
Note: HuggingFace cache also goes to scratch to avoid filling home dir — set HF_HOME=$HOME/scratch/hf_cache before downloading models. The pace_train_llama.sh script does this automatically.
Note: if home dir fills up, run `rm -rf $HOME/.cache/huggingface` to clear cached model weights from home.

# HuggingFace token setup (required for gated models e.g. Llama):
- Create an account at huggingface.co
- Go to huggingface.co/settings/tokens and create a token with Read permissions
- Accept the model license at huggingface.co/meta-llama/Llama-3.1-8B
- Log in locally: `conda activate <your-env>` then `hf auth login` and paste your token

# Prep the data:
5. run the 'Generating splits before training' steps below to create splits from inferred labels

### Generating splits before training

Before running `pace_train.sh` or any training script, you need to generate the train/dev/test splits from the inferred labels. Use `prepare_splits_from_labels.py` (run from repo root):

```bash
# Combine all eras into single files
cat ~/scratch/data/annotations/relevance_and_tone/inferred_labels/early_relevance_all.jsonlist \
    data/annotations/relevance_and_tone/inferred_labels/mid_relevance_all.jsonlist \
    data/annotations/relevance_and_tone/inferred_labels/modern_relevance_all.jsonlist \
    > /tmp/all_relevance.jsonlist

cat ~/scratch/annotations/relevance_and_tone/inferred_labels/early_tone_all.jsonlist \
    data/annotations/relevance_and_tone/inferred_labels/mid_tone_all.jsonlist \
    data/annotations/relevance_and_tone/inferred_labels/modern_tone_all.jsonlist \
    > /tmp/all_tone.jsonlist

# Generate relevance splits
python prepare_splits_from_labels.py /tmp/all_relevance.jsonlist \
    --basedir ~/scratch/data/speeches/Congress/relevance/splits/basic/

# Generate tone splits
python prepare_splits_from_labels.py /tmp/all_tone.jsonlist \
    --basedir ~/scratch/data/speeches/Congress/tone/splits/label-weights/
```

This creates `all.jsonlist` and `folds/{0..9}/train|dev|test.jsonlist` under each basedir, which is what `pace_train.sh` and the classification scripts expect.


# Experiment 1:
6. make .sh file with instructions for training

Experiment 1's training script is in pace_train.sh. run the training via sbatch <name of file>

# Experiment 2:


# Experiment 3: Temporal Generalization

Trains one model per historical era (early / mid / modern) and evaluates each on all three eras' test sets, producing a 3×3 grid of macro-F1 scores. All 9 combinations are handled by a single script.

**Prerequisite:** per-era splits must exist under `data/annotations/relevance_and_tone/{era}/relevance/splits/basic/`. Run the split generation step from "Prep the data" section above if needed.

### Quick local test (MPS/CPU, fast)

```bash
# Run from repo root: us-immigration-speeches-1.1/
python -m classification.run_temporal_generalization \
  --model_type distilbert \
  --model_name_or_path distilbert-base-uncased \
  --split basic \
  --n_epochs 2 \
  --per_gpu 8 \
  --seed 42
```

### Full run on PACE (GPU, 7 epochs)

```bash
sbatch pace_temporal_gen.sh
```

Edit `pace_temporal_gen.sh` to change the model or hyperparameters.

### Eval-only (if checkpoints already exist)

```bash
python -m classification.run_temporal_generalization \
  --model_type distilbert \
  --model_name_or_path distilbert-base-uncased \
  --split basic \
  --seed 42 \
  --eval-only
```

### Output

Results are printed as a 3×3 table and saved to:
```
results/temporal_generalization/{model}_s{seed}/summary.tsv
```

- Diagonal entries = within-era (in-distribution) performance
- Off-diagonal entries = cross-era transfer
- Trained checkpoints are saved under each era's split dir:
  `data/annotations/relevance_and_tone/{era}/relevance/splits/basic/bert/temporal_{era}_{model}_s{seed}_lr{lr}_msl{msl}/`
- The script skips re-training if a valid checkpoint exists; use `--overwrite` to force re-training
