"""
Prepare train/dev/test splits and cross-validation folds from a combined
inferred-labels jsonlist file (format produced by the label aggregation pipeline).

Usage:
  python prepare_splits_from_labels.py <all.jsonlist> --basedir <output_dir>

Each line of the input jsonlist must have: id, text, tokens, label, phase, weight

Outputs (mirrors the structure expected by run_final_model.py and run_folds_hf.py):
  <basedir>/all.jsonlist          -- full dataset (already exists; symlinked/copied)
  <basedir>/folds/<n>/train.jsonlist
  <basedir>/folds/<n>/dev.jsonlist
  <basedir>/folds/<n>/test.jsonlist
"""

import json
import os
import random
import numpy as np
from optparse import OptionParser
from collections import defaultdict


def main():
    usage = "%prog all.jsonlist"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='splits',
                      help='Base output directory: default=%default')
    parser.add_option('--folds', type=int, default=10,
                      help='Number of cross-validation folds: default=%default')
    parser.add_option('--dev-frac', type=float, default=0.1,
                      help='Fraction of data for dev set per fold: default=%default')
    parser.add_option('--test-frac', type=float, default=0.1,
                      help='Fraction of data for test set per fold: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.error("Must provide path to all.jsonlist")

    infile = args[0]
    basedir = options.basedir
    n_folds = options.folds
    dev_frac = options.dev_frac
    test_frac = options.test_frac
    seed = options.seed

    random.seed(seed)
    np.random.seed(seed)

    # Load data
    items = []
    with open(infile) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"Loaded {len(items)} items from {infile}")

    # Summarize label distribution
    label_counts = defaultdict(int)
    for item in items:
        label_counts[item['label']] += 1
    print(f"Label distribution: {dict(label_counts)}")

    # Copy all.jsonlist to basedir if not already there
    os.makedirs(basedir, exist_ok=True)
    all_out = os.path.join(basedir, 'all.jsonlist')
    if not os.path.exists(all_out):
        with open(all_out, 'w') as f:
            for item in items:
                f.write(json.dumps(item) + '\n')
        print(f"Written {all_out}")

    # Shuffle items
    indices = list(range(len(items)))
    random.shuffle(indices)

    n = len(items)
    n_test = max(1, int(n * test_frac))
    n_dev = max(1, int(n * dev_frac))

    # --- Single simple split (train/dev/test) ---
    test_idx = set(indices[:n_test])
    dev_idx = set(indices[n_test:n_test + n_dev])
    train_idx = set(indices[n_test + n_dev:])

    write_split(basedir, 'train', [items[i] for i in sorted(train_idx)])
    write_split(basedir, 'dev',   [items[i] for i in sorted(dev_idx)])
    write_split(basedir, 'test',  [items[i] for i in sorted(test_idx)])
    print(f"Simple split: train={len(train_idx)}, dev={len(dev_idx)}, test={len(test_idx)}")

    # --- Cross-validation folds ---
    fold_size = n // n_folds
    for fold in range(n_folds):
        fold_dir = os.path.join(basedir, 'folds', str(fold))
        os.makedirs(fold_dir, exist_ok=True)

        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        fold_test_idx = set(indices[test_start:test_end])

        remaining = [i for i in indices if i not in fold_test_idx]
        n_fold_dev = max(1, int(len(remaining) * dev_frac))
        fold_dev_idx = set(remaining[:n_fold_dev])
        fold_train_idx = set(remaining[n_fold_dev:])

        write_split(fold_dir, 'train', [items[i] for i in sorted(fold_train_idx)])
        write_split(fold_dir, 'dev',   [items[i] for i in sorted(fold_dev_idx)])
        write_split(fold_dir, 'test',  [items[i] for i in sorted(fold_test_idx)])

    print(f"Created {n_folds} CV folds in {os.path.join(basedir, 'folds')}/")


def write_split(outdir, prefix, items):
    outfile = os.path.join(outdir, f'{prefix}.jsonlist')
    with open(outfile, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    main()
