"""
classification/run_temporal_generalization.py

Experiment 3: Temporal Generalization
Train one model per historical era (early / mid / modern) and evaluate each
model on all three eras' test sets, producing a 3×3 grid of macro-averaged F1
scores (9 values total).

Usage:
  python -m classification.run_temporal_generalization \
      --model_type roberta \
      --model_name_or_path roberta-base \
      --split basic \
      --n_epochs 7 \
      --lr 2e-5 \
      --per_gpu 4 \
      --seed 42

Results are written to:
  results/temporal_generalization/{model_short}_s{seed}/
    train_{train_era}_test_{test_era}/   ← hf.run output for each pair
  results/temporal_generalization/{model_short}_s{seed}/summary.tsv
"""

import os
import sys
import json
import logging
from subprocess import call
from optparse import OptionParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

ERAS = ["early", "mid", "modern"]

# Base dir (used for checkpoint output path)
SPLIT_DIR_TMPL = "data/annotations/relevance_and_tone/{era}/relevance/splits/{split}/"

# nontest/ subdir contains nontest.jsonlist (all non-test data)
TRAIN_DIR_TMPL = "data/annotations/relevance_and_tone/{era}/relevance/splits/{split}/nontest/"

# Canonical test split lives in fold 0
TEST_DIR_TMPL = "data/annotations/relevance_and_tone/{era}/relevance/splits/{split}/folds/0/"

TRAIN_FILE = "nontest.jsonlist"  # all non-test data; for training & label-list derivation
TEST_FILE = "test.jsonlist"      # canonical held-out test set (same across all folds)


def era_split_dir(era: str, split: str) -> str:
    """Base dir — used for checkpoint output."""
    return SPLIT_DIR_TMPL.format(era=era, split=split)


def era_train_dir(era: str, split: str) -> str:
    """Dir that contains nontest.jsonlist."""
    return TRAIN_DIR_TMPL.format(era=era, split=split)


def era_test_dir(era: str, split: str) -> str:
    """Dir that contains test.jsonlist."""
    return TEST_DIR_TMPL.format(era=era, split=split)


def model_output_dir(basedir: str, model_name_or_path: str, seed: int,
                     lr: float, max_seq_length: int,
                     output_prefix: str = "temporal") -> str:
    """Mirrors the path constructed by run_final_model.run()."""
    model_short = os.path.basename(model_name_or_path.rstrip("/"))
    prefix = f"{output_prefix}_{model_short}_s{seed}_lr{lr}_msl{max_seq_length}"
    return os.path.join(basedir, "bert", prefix)


def run_training(train_era: str, opts) -> str:
    """Train on one era and return the path to the saved model checkpoint."""
    basedir = era_split_dir(train_era, opts.split)
    outdir = model_output_dir(
        basedir, opts.model_name_or_path, opts.seed, opts.lr,
        opts.max_seq_length, output_prefix=f"temporal_{train_era}",
    )

    has_model = os.path.isdir(outdir) and any(
        f in os.listdir(outdir) for f in ("pytorch_model.bin", "model.safetensors", "config.json")
    )
    if has_model and not opts.overwrite:
        logger.info(f"[train/{train_era}] Checkpoint already exists at {outdir} — skipping training.")
        return outdir

    train_dir = era_train_dir(train_era, opts.split)
    name = os.path.basename(basedir.rstrip("/"))
    cmd = [
        sys.executable, "-m", "hf.run",
        "--model_type", opts.model_type,
        "--model_name_or_path", opts.model_name_or_path,
        "--name", name,
        "--do_train",
        "--train", TRAIN_FILE,
        "--data_dir", train_dir,
        "--max_seq_length", str(opts.max_seq_length),
        f"--per_gpu_train_batch_size={opts.per_gpu}",
        f"--per_gpu_eval_batch_size={opts.per_gpu}",
        "--learning_rate", str(opts.lr),
        "--num_train_epochs", str(opts.n_epochs),
        "--output_dir", outdir,
        "--overwrite_cache",
        "--overwrite_output_dir",
        "--weight_field", "weight",
        "--metrics", "accuracy,f1",
        "--seed", str(opts.seed),
    ]
    if opts.model_type == "bert":
        cmd.append("--do_lower_case")

    logger.info(f"[train/{train_era}] " + " ".join(cmd))
    call(cmd)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "train_cmd.txt"), "w") as f:
        f.write(" ".join(cmd) + "\n")

    return outdir


def run_cross_eval(train_era: str, test_era: str,
                   checkpoint_dir: str, opts, results_root: str) -> dict:
    """
    Evaluate a trained model (from train_era) on test_era's test split.
    Returns the parsed eval metrics dict, or {} on failure.
    """
    train_data_dir = era_train_dir(train_era, opts.split)
    test_dir = era_test_dir(test_era, opts.split)
    outdir = os.path.join(results_root, f"train_{train_era}_test_{test_era}")
    os.makedirs(outdir, exist_ok=True)

    name = os.path.basename(era_split_dir(train_era, opts.split).rstrip("/"))
    cmd = [
        sys.executable, "-m", "hf.run",
        "--model_type", opts.model_type,
        "--model_name_or_path", checkpoint_dir,
        "--name", name,
        "--do_eval",
        "--eval_partition", "test",
        # data_dir points to nontest/ so hf.run can derive the label list
        "--data_dir", train_data_dir,
        "--train", TRAIN_FILE,
        # test examples come from the (possibly different) test era's fold-0 dir
        "--test_data_dir", test_dir,
        "--test", TEST_FILE,
        "--max_seq_length", str(opts.max_seq_length),
        f"--per_gpu_eval_batch_size={opts.per_gpu}",
        "--output_dir", outdir,
        "--overwrite_cache",
        "--overwrite_output_dir",
        "--metrics", "accuracy,f1",
        "--seed", str(opts.seed),
    ]
    if opts.model_type == "bert":
        cmd.append("--do_lower_case")

    logger.info(f"[eval/{train_era}→{test_era}] " + " ".join(cmd))
    call(cmd)

    # Parse results written by hf.run (eval_results.txt or similar)
    return parse_eval_results(outdir)


def parse_eval_results(outdir: str) -> dict:
    """
    Read the eval_results.txt file written by hf.run and return a dict of
    metric→value pairs. Returns {} if no results file is found.
    """
    for fname in ("eval_results.txt", "eval_results_test.txt"):
        fpath = os.path.join(outdir, fname)
        if os.path.isfile(fpath):
            metrics = {}
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line:
                        key, _, val = line.partition("=")
                        try:
                            metrics[key.strip()] = float(val.strip())
                        except ValueError:
                            pass
            return metrics
    logger.warning(f"No eval_results file found in {outdir}")
    return {}


def print_summary(results: dict):
    """Print and return the 3×3 macro-F1 grid."""
    print("\n=== Temporal Generalization: macro-F1 ===")
    header = "train\\test".ljust(12) + "  ".join(e.ljust(8) for e in ERAS)
    print(header)
    print("-" * len(header))
    for train_era in ERAS:
        row = train_era.ljust(12)
        for test_era in ERAS:
            m = results.get((train_era, test_era), {})
            # hf/run.py writes "eval_f1" or "f1"; try both
            f1 = m.get("eval_f1", m.get("f1", float("nan")))
            row += f"{f1:.4f}".ljust(10)
        print(row)
    print()


def main():
    parser = OptionParser(usage="%prog [options]")
    parser.add_option("--model_type", type=str, default="roberta",
                      help="Model type: [bert|roberta|distilbert|...] default=%default")
    parser.add_option("--model_name_or_path", type=str, default="roberta-base",
                      help="HuggingFace model id or local path: default=%default")
    parser.add_option("--split", type=str, default="basic",
                      help="Split variant to use [basic|label-weights]: default=%default")
    parser.add_option("--n_epochs", type=int, default=7,
                      help="Training epochs: default=%default")
    parser.add_option("--lr", type=float, default=2e-5,
                      help="Learning rate: default=%default")
    parser.add_option("--per_gpu", type=int, default=4,
                      help="Batch size per GPU: default=%default")
    parser.add_option("--max_seq_length", type=int, default=512,
                      help="Max sequence length: default=%default")
    parser.add_option("--seed", type=int, default=42,
                      help="Random seed: default=%default")
    parser.add_option("--results-dir", type=str,
                      default="results/temporal_generalization",
                      help="Root directory for all results: default=%default")
    parser.add_option("--overwrite", action="store_true", default=False,
                      help="Re-train even if a checkpoint already exists")
    parser.add_option("--eval-only", action="store_true", default=False,
                      help="Skip training; run cross-era eval using existing checkpoints")

    (opts, _) = parser.parse_args()

    model_short = os.path.basename(opts.model_name_or_path.rstrip("/"))
    results_root = os.path.join(
        opts.results_dir, f"{model_short}_s{opts.seed}"
    )
    os.makedirs(results_root, exist_ok=True)

    # ── Step 1: Train one model per era ──────────────────────────────────────
    checkpoints = {}   # era → checkpoint dir
    for train_era in ERAS:
        if opts.eval_only:
            # Reconstruct the expected checkpoint path without re-training
            basedir = era_split_dir(train_era, opts.split)
            ckpt = model_output_dir(
                basedir, opts.model_name_or_path, opts.seed, opts.lr,
                opts.max_seq_length, output_prefix=f"temporal_{train_era}",
            )
            logger.info(f"[eval-only/{train_era}] Using checkpoint: {ckpt}")
            has_model = os.path.isdir(ckpt) and any(
                f in os.listdir(ckpt) for f in ("pytorch_model.bin", "model.safetensors", "config.json")
            )
            if not has_model:
                logger.error(
                    f"--eval-only set but no model files found in: {ckpt}\n"
                    f"Run without --eval-only to train first."
                )
                sys.exit(1)
            checkpoints[train_era] = ckpt
        else:
            checkpoints[train_era] = run_training(train_era, opts)

    # ── Step 2: Evaluate all 9 train×test combinations ───────────────────────
    results = {}
    for train_era in ERAS:
        for test_era in ERAS:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: train={train_era}  test={test_era}")
            logger.info(f"{'='*60}")
            metrics = run_cross_eval(
                train_era, test_era,
                checkpoints[train_era], opts, results_root,
            )
            results[(train_era, test_era)] = metrics

    # ── Step 3: Print the 3×3 grid and save summary ──────────────────────────
    print_summary(results)

    # Write TSV summary
    summary_path = os.path.join(results_root, "summary.tsv")
    with open(summary_path, "w") as f:
        f.write("train_era\ttest_era\t" + "\t".join(
            sorted(next(iter(results.values()), {}).keys())
        ) + "\n")
        for train_era in ERAS:
            for test_era in ERAS:
                m = results.get((train_era, test_era), {})
                row = f"{train_era}\t{test_era}\t" + "\t".join(
                    str(m.get(k, "")) for k in sorted(m.keys())
                )
                f.write(row + "\n")
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
