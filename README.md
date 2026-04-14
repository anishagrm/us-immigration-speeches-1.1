
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
Note: HuggingFace cache also goes to scratch to avoid filling home dir — set HF_HOME=$HOME/scratch/hf_cache before downloading models. The llama-runs/run1/pace_train_llama_tone.sh and llama-runs/run1/pace_train_llama_relevance.sh scripts do this automatically.
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

# Experiment 2: Fine-tuned Llama (QLoRA)

Fine-tunes Llama-3.2-1B on the annotated corpus using QLoRA (4-bit NF4 + LoRA rank-16/alpha-16 on q_proj, v_proj). Unlike the encoder-based experiments, Llama receives the annotator guidelines verbatim in its prompt and classifies by comparing log-probabilities over label tokens — no classification head needed.

Two tasks are supported: **tone** (anti / neutral / pro) and **relevance** (yes / no).

### Environment setup

Uses the `llama` conda env (Python 3.11), not `hum`. See the environment setup section at the top.

Conda env packages needed:
```bash
pip install "transformers>=4.40" "peft>=0.10" bitsandbytes accelerate pandas numpy tqdm scikit-learn huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Data

**Tone:** `data/speeches/Congress/tone/splits/label-weights/`  
**Relevance:** `data/speeches/Congress/relevance/splits/basic/`

Each directory needs `all.jsonlist` (train), `dev.jsonlist`, and `test.jsonlist`. Generate these with the split generation step in "Prep the data" if missing.

### Training on PACE

```bash
# Tone classifier (Llama-3.2-1B, 3 epochs, QLoRA)
sbatch llama-runs/run1/pace_train_llama_tone.sh

# Relevance classifier (Llama-3.2-1B, 3 epochs, QLoRA)
sbatch llama-runs/run1/pace_train_llama_relevance.sh
```

Both scripts:
- Download and cache the model to `$HOME/scratch/hf_cache` before training
- Use batch size 4, gradient accumulation 4 (effective batch 16), lr 2e-4, cosine schedule
- Save the fine-tuned LoRA adapter to `data/speeches/Congress/{task}/splits/{split}/llama/llama_qlora_Llama-3.2-1B_s42_lr0.0002/`

### Running manually

```bash
# Tone
python -m classification.run_llama_qlora \
    --model meta-llama/Llama-3.2-1B \
    --basedir data/speeches/Congress/tone/splits/label-weights/ \
    --train-file all.jsonlist --dev-file dev.jsonlist --test-file test.jsonlist \
    --output-prefix llama_qlora \
    --do-train --do-eval \
    --n-epochs 3 --lr 2e-4 --batch-size 4 --grad-accum 4 \
    --lora-rank 16 --lora-alpha 16 --max-seq-length 512 --seed 42

# Relevance
python -m classification.run_llama_qlora_relevance \
    --model meta-llama/Llama-3.2-1B \
    --basedir data/speeches/Congress/relevance/splits/basic/ \
    --train-file all.jsonlist --dev-file dev.jsonlist --test-file test.jsonlist \
    --output-prefix llama_qlora \
    --do-train --do-eval \
    --n-epochs 3 --lr 2e-4 --batch-size 4 --grad-accum 4 \
    --lora-rank 16 --lora-alpha 16 --max-seq-length 512 --seed 42
```

### Eval-only from saved checkpoint

```bash
python -m classification.run_llama_qlora \
    --model meta-llama/Llama-3.2-1B \
    --checkpoint data/speeches/Congress/tone/splits/label-weights/llama/llama_qlora_Llama-3.2-1B_s42_lr0.0002 \
    --basedir data/speeches/Congress/tone/splits/label-weights/ \
    --dev-file dev.jsonlist --test-file test.jsonlist \
    --do-eval --seed 42
```

### Output

Predictions and metrics are written to the output dir:
- `preds_dev.tsv` / `preds_test.tsv` — predicted label indices and class probabilities
- `eval_results_dev.txt` / `eval_results_test.txt` — accuracy, per-class F1, macro-F1

### Notes

- 4-bit quantization requires CUDA; MPS/CPU automatically fall back to fp16/fp32 full precision
- Classification is done via log-probability comparison over label tokens at the last prompt position — no additional classification head
- Tone label tokens: `negative` (anti), `neutral`, `positive` (pro)
- Relevance label tokens: `no`, `yes`


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
