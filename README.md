
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

### Bugs fixed (run2)

Run1 produced loss=0 throughout training due to two bugs in the dataset construction, both in `ToneDataset` / `RelevanceDataset`:

1. **Label token truncated by long speeches.** The full sequence (guidelines + speech + label) was tokenized with `truncation=True, max_length=512`, which cuts from the right — silently removing the label token for any speech long enough to fill the context. Fix: pre-compute the token budget for the speech text (`max_length - template_tokens - 1`) and truncate only the speech, guaranteeing the label token always survives.

2. **Tokenization boundary merge.** Even after fix #1, loss was still 0. The trailing `"Tone: "` produces a standalone space token when tokenized alone, but when the label is appended (`"Tone: negative"`), the tokenizer merges the space into the label token (`"▁negative"`). This made `len(full_ids) == len(prompt_ids)`, so `labels` was all `-100`. Fix: instead of using `len(prompt_ids)` as the boundary, find the first position where `full_ids` and `prompt_ids` diverge — this correctly handles both clean appends and boundary merges.

A `ZeroLossCallback` was also added to both scripts: if loss stays exactly `0.0` for 5 consecutive logging steps, training aborts immediately with an error rather than wasting GPU hours.

**Use run2 scripts, not run1.**

### Training on PACE

```bash
# Tone classifier (Llama-3.2-1B, 3 epochs, QLoRA)
sbatch llama-runs/run2/pace_train_llama_tone.sh

# Relevance classifier (Llama-3.2-1B, 3 epochs, QLoRA)
sbatch llama-runs/run2/pace_train_llama_relevance.sh
```

Both scripts:
- Download and cache the model to `$HOME/scratch/hf_cache` before training
- Use batch size 4, gradient accumulation 4 (effective batch 16), lr 2e-4, cosine schedule
- Save results to `llama-runs/run2/tone/` and `llama-runs/run2/relevance/`

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

### Results

#### Run1 (broken — loss=0 throughout)

| Task | Macro-F1 (dev) | Macro-F1 (test) |
|------|---------------|-----------------|
| Tone | 0.32 | 0.32 |
| Relevance | 0.50 | 0.49 |

Both at random-chance level (3-class baseline = 0.33, binary baseline = 0.50). Training loss was 0 for all steps due to the bugs described above — the model never learned anything.

#### Run2 (bugs fixed)

| Task | Metric | Dev | Test |
|------|--------|-----|------|
| Tone | Macro-F1 | 0.4153 | 0.3985 |
| Tone | F1-anti | 0.3852 | 0.4065 |
| Tone | F1-neutral | 0.4242 | 0.4505 |
| Tone | F1-pro | 0.4364 | 0.3386 |
| Relevance | Macro-F1 | 0.5174 | 0.4874 |
| Relevance | F1-yes | 0.4028 | 0.3563 |
| Relevance | Accuracy | 0.5446 | 0.5210 |

**Comparison to run1:** Tone macro-F1 improved from 0.32 → 0.40 (+8 points), confirming the model is now actually learning. Relevance macro-F1 improved marginally (0.49 → 0.49) at the macro level but F1-yes improved (0.35 → 0.36), again showing real learning.

**Interpretation:** Tone results are meaningfully above chance and show the model has learned to distinguish anti/neutral/pro framing. Neutral is the easiest class (F1=0.45), which makes sense as it has the most distinct surface features (procedural language, statistics). Pro and anti are harder to separate. Relevance results are weaker — the model is still closer to the binary baseline (0.50), suggesting the 1B model struggles with the relevance boundary more than tone. Increasing to a larger model or more epochs would likely help both tasks.

### Notes

- 4-bit quantization requires CUDA; MPS/CPU automatically fall back to fp32
- Classification is done via log-probability comparison over label tokens at the last prompt position — no additional classification head
- Tone label tokens: `negative` (anti), `neutral`, `positive` (pro)
- Relevance label tokens: `no`, `yes`


# Experiment 3: Temporal Generalization

### What the model does

This is encoder-only fine-tuning — no prompt, no generation. RoBERTa-base is fine-tuned for binary classification: given a raw congressional speech segment, predict `yes` (relevant to immigration) or `no` (not relevant). A linear classification head is added on top of RoBERTa's `[CLS]` token representation and trained jointly with the encoder weights via cross-entropy loss on labeled examples. The model sees only the speech text — no task description or instructions.

Three separate models are trained, one per historical era:
- **Early** (~1870s–1920s): Reconstruction, Chinese Exclusion Act, first great immigration wave
- **Mid** (~1930s–1970s): New Deal through Cold War, national quota system debates
- **Modern** (~1980s–2010s): Reagan-era reform, post-9/11, contemporary border policy

Each model learns era-specific surface features that correlate with immigration relevance (e.g. the modern model picks up on "undocumented", "DACA", "border"; the early model on "Chinese", "alien", "naturalization"). The experiment then tests whether those features transfer across time by evaluating all three models on all three test sets — producing a 3×3 grid. Cross-era drops reveal how much immigration rhetoric has shifted: a modern model that learned contemporary vocabulary struggles to recognize an 1880s speech about the Chinese Exclusion Act as immigration-relevant, even though it clearly is.

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

### Results (RoBERTa-base, seed 42, 3 epochs)

Macro-F1 scores for all 9 train-era × test-era combinations on the relevance task (is this speech about immigration?).

| Train \ Test | Early  | Mid    | Modern |
|--------------|--------|--------|--------|
| **Early**    | 0.9603 | 0.8542 | 0.8000 |
| **Mid**      | 0.8624 | 0.9722 | 0.8250 |
| **Modern**   | 0.8148 | 0.9028 | 0.9583 |

**Diagonal** = within-era (in-distribution) performance — all ≥ 0.96, confirming each era is learnable.

**Off-diagonal** = cross-era transfer. Key observations:
- Early → Modern and Modern → Early show the largest drops (~0.80 and ~0.81), consistent with the greatest temporal distance.
- Mid transfers reasonably well in both directions (0.86–0.90), acting as a bridge era.
- Overall the drops are modest (~10–15 F1 points), suggesting immigration rhetoric shares substantial cross-era signal — but era-specific features still matter.
