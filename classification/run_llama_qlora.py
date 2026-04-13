"""
classification/run_llama_qlora.py

Fine-tune meta-llama/Llama-3.1-8B for immigration-speech tone classification
using QLoRA (4-bit NF4 quantization + LoRA rank-16/alpha-16 on q_proj, v_proj).

A novel aspect vs. the encoder experiments: the annotator guidelines are embedded
verbatim in the prompt, so the model is supervised by a natural-language task
description rather than just examples.

At inference, tone is determined by comparing the log-probabilities the model
assigns to the three label tokens ("negative", "neutral", "positive") given
the formatted input prompt — no classification head needed.

Data labels:  anti | neutral | pro
Prompt tokens: negative | neutral | positive
Output TSV columns match hf/run.py: true, predicted, 0, 1, 2
(sorted label order: anti=0, neutral=1, pro=2)

Requirements: transformers>=4.40, peft>=0.10, bitsandbytes>=0.43 (CUDA only),
              torch, pandas, numpy, tqdm, scikit-learn

Usage — train then eval:
  python -m classification.run_llama_qlora \\
      --basedir data/speeches/Congress/tone/splits/label-weights/ \\
      --train-file all.jsonlist \\
      --dev-file dev.jsonlist \\
      --test-file test.jsonlist \\
      --output-prefix llama_qlora \\
      --do-train --do-eval \\
      --n-epochs 3 --seed 42

Usage — eval only from a saved checkpoint:
  python -m classification.run_llama_qlora \\
      --basedir data/speeches/Congress/tone/splits/label-weights/ \\
      --checkpoint path/to/saved/model \\
      --do-eval
"""

import os
import json
import random
import logging
from functools import partial
from optparse import OptionParser

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Label constants ──────────────────────────────────────────────────────────
# Data file values → sorted alphabetically → matches hf/run.py index order
SORTED_LABELS = ["anti", "neutral", "pro"]   # indices 0, 1, 2
LABEL2IDX = {l: i for i, l in enumerate(SORTED_LABELS)}

# Natural-language tokens used inside the prompt for each data label
LABEL2TOKEN = {"anti": "negative", "neutral": "neutral", "pro": "positive"}

# ── Annotator guidelines (verbatim in prompt) ────────────────────────────────
ANNOTATOR_GUIDELINES = """\
You are an expert annotator of political speech. Your task is to classify the \
tone of a Congressional speech segment about immigration.

ANNOTATION GUIDELINES
---------------------
positive  The segment frames immigration or immigrants favorably. This includes
          emphasizing economic contributions, humanitarian values, cultural
          enrichment, legal rights, or support for inclusive immigration policy.
          The speaker expresses approval of immigration or solidarity with
          immigrants.

negative  The segment frames immigration or immigrants unfavorably. This includes
          emphasizing threats to jobs, national security, cultural cohesion, or
          the rule of law, or advocacy for restrictive immigration measures. The
          speaker expresses opposition to immigration or concern about its effects.

neutral   The segment is factual, procedural, or administrative in nature and does
          not clearly express a positive or negative stance. This includes
          recitation of statistics, descriptions of pending legislation without
          advocacy, or procedural floor debate.

Respond with exactly one word on a new line: positive, negative, or neutral.
"""


def build_prompt(text: str, label: str = None) -> str:
    """
    Build the full prompt for a single example.

    The prompt ends with 'Tone: ' so the model's next token is the label.
    If `label` is provided (for training), the label token is appended.
    """
    prompt = (
        f"{ANNOTATOR_GUIDELINES}\n"
        f"Speech segment:\n\"{text}\"\n\n"
        "Tone: "
    )
    if label is not None:
        prompt += LABEL2TOKEN[label]
    return prompt


# ── Data loading ─────────────────────────────────────────────────────────────

def load_jsonlist(path: str):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ── Dataset ───────────────────────────────────────────────────────────────────

class ToneDataset(Dataset):
    """
    Tokenises speech segments with the annotator-guideline prompt for SFT.
    The loss is computed only on the label token; all prompt tokens are masked
    with -100 in the labels tensor.
    """

    def __init__(self, items, tokenizer, max_length: int = 512):
        self.examples = []
        for item in items:
            text = item["text"]
            label = item["label"]

            full_prompt = build_prompt(text, label=label)
            prompt_only = build_prompt(text, label=None)

            full_enc = tokenizer(
                full_prompt, max_length=max_length, truncation=True
            )
            prompt_len = len(
                tokenizer(prompt_only, max_length=max_length, truncation=True)[
                    "input_ids"
                ]
            )

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            # Mask prompt tokens from loss; supervise only the label token(s)
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {k: torch.tensor(v) for k, v in ex.items()}


def collate_fn(batch, pad_token_id: int):
    """Left-pad a batch of variable-length sequences."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids_out, attn_out, labels_out = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids_out.append(
            torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long),
                       x["input_ids"]])
        )
        attn_out.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long),
                       x["attention_mask"]])
        )
        labels_out.append(
            torch.cat([torch.full((pad_len,), -100, dtype=torch.long),
                       x["labels"]])
        )
    return {
        "input_ids": torch.stack(input_ids_out),
        "attention_mask": torch.stack(attn_out),
        "labels": torch.stack(labels_out),
    }


# ── Inference helpers ─────────────────────────────────────────────────────────

def get_label_token_ids(tokenizer) -> list:
    """
    Return the single token IDs for 'negative', 'neutral', 'positive' as they
    appear immediately after 'Tone: ' in the model's vocabulary.

    Returns a list of three IDs in SORTED_LABELS order:
      [id_for_negative, id_for_neutral, id_for_positive]
      = [id_for_anti,   id_for_neutral, id_for_pro]
    """
    tokens = [LABEL2TOKEN[l] for l in SORTED_LABELS]  # ['negative','neutral','positive']
    ids = []
    for tok in tokens:
        # Try without space, with space, and as a continuation token
        for candidate in [tok, " " + tok]:
            enc = tokenizer.encode(candidate, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
                break
        else:
            enc = tokenizer.encode(tok, add_special_tokens=False)
            ids.append(enc[0])
            logger.warning(
                f"Label token '{tok}' is multi-subword ({enc}); "
                f"using first subtoken {enc[0]}"
            )
    logger.info(
        "Label token IDs  "
        + "  ".join(f"{l}({LABEL2TOKEN[l]})={i}" for l, i in zip(SORTED_LABELS, ids))
    )
    return ids


@torch.no_grad()
def score_examples(model, tokenizer, items, label_token_ids, device,
                   batch_size: int = 8, max_length: int = 512):
    """
    For each item, extract log-probabilities the model assigns to the three
    label tokens given the prompt, then softmax over those three values.

    Returns:
      true_indices  np.ndarray (N,)  gold label indices (-1 if unlabelled)
      predicted     np.ndarray (N,)  argmax of probs
      probs_matrix  np.ndarray (N,3) columns: [P(anti), P(neutral), P(pro)]
    """
    model.eval()
    true_indices, all_probs = [], []

    for start in tqdm(range(0, len(items), batch_size), desc="Scoring"):
        batch = items[start : start + batch_size]
        prompts = [build_prompt(item["text"]) for item in batch]

        enc = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = model(**enc)
        logits = outputs.logits  # (B, T, V)

        # The label token is predicted at the last non-padding position
        seq_lens = enc["attention_mask"].sum(dim=1) - 1  # (B,)

        for i, item in enumerate(batch):
            last_pos = seq_lens[i].item()
            logit_vec = logits[i, last_pos, :]      # (V,)
            label_logits = logit_vec[label_token_ids]  # (3,) one per class
            probs = torch.softmax(label_logits, dim=0).cpu().float().numpy()
            all_probs.append(probs)
            true_indices.append(LABEL2IDX.get(item.get("label", ""), -1))

    probs_matrix = np.array(all_probs)           # (N, 3)
    predicted = np.argmax(probs_matrix, axis=1)  # (N,)
    return np.array(true_indices), predicted, probs_matrix


def save_predictions(outdir: str, split_name: str,
                     true_indices, predicted, probs_matrix):
    """
    Save TSV in the same format as hf/run.py evaluate():
      columns: true, predicted, 0, 1, 2

    eval_results_{split}.txt uses the same key=value format as hf/run.py so
    results are directly comparable across experiments:
      acc = 0.xxxx
      f1-anti = 0.xxxx
      f1-neutral = 0.xxxx
      f1-pro = 0.xxxx
      macro_f1 = 0.xxxx
    """
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame({"true": true_indices, "predicted": predicted})
    for i in range(probs_matrix.shape[1]):
        df[i] = probs_matrix[:, i]
    outpath = os.path.join(outdir, f"preds_{split_name}.tsv")
    df.to_csv(outpath, sep="\t", index=False)
    logger.info(f"Saved {len(df)} predictions → {outpath}")

    valid = true_indices >= 0
    if valid.sum() == 0:
        return

    from sklearn.metrics import accuracy_score, f1_score
    t, p = true_indices[valid], predicted[valid]
    acc = accuracy_score(t, p)
    per_class_f1 = f1_score(t, p, labels=list(range(len(SORTED_LABELS))),
                            average=None, zero_division=0)
    macro_f1 = f1_score(t, p, average="macro", zero_division=0)

    lines = [f"acc = {acc:.4f}"]
    for label, f1 in zip(SORTED_LABELS, per_class_f1):
        lines.append(f"f1-{label} = {f1:.4f}")
    lines.append(f"macro_f1 = {macro_f1:.4f}")
    result_str = "\n".join(lines)

    logger.info(f"\n=== {split_name} ===\n{result_str}")
    with open(os.path.join(outdir, f"eval_results_{split_name}.txt"), "w") as f:
        f.write(result_str + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option("--model", type=str, default="meta-llama/Llama-3.1-8B",
                      help="HuggingFace model name or path: default=%default")
    parser.add_option("--checkpoint", type=str, default=None,
                      help="Path to a saved PEFT checkpoint to load for eval-only")
    parser.add_option("--basedir", type=str,
                      default="data/speeches/Congress/tone/splits/label-weights/",
                      help="Base data directory: default=%default")
    parser.add_option("--train-file", type=str, default="all.jsonlist",
                      help="Training filename inside basedir: default=%default")
    parser.add_option("--dev-file", type=str, default="dev.jsonlist",
                      help="Dev filename inside basedir: default=%default")
    parser.add_option("--test-file", type=str, default="test.jsonlist",
                      help="Test filename inside basedir: default=%default")
    parser.add_option("--output-prefix", type=str, default="llama_qlora",
                      help="Prefix for the output subdirectory: default=%default")
    parser.add_option("--n-epochs", type=int, default=3,
                      help="Training epochs: default=%default")
    parser.add_option("--lr", type=float, default=2e-4,
                      help="Learning rate: default=%default")
    parser.add_option("--batch-size", type=int, default=4,
                      help="Per-device train batch size: default=%default")
    parser.add_option("--grad-accum", type=int, default=4,
                      help="Gradient accumulation steps: default=%default")
    parser.add_option("--max-seq-length", type=int, default=512,
                      help="Max sequence length: default=%default")
    parser.add_option("--lora-rank", type=int, default=16,
                      help="LoRA rank: default=%default")
    parser.add_option("--lora-alpha", type=int, default=16,
                      help="LoRA alpha: default=%default")
    parser.add_option("--eval-batch-size", type=int, default=8,
                      help="Batch size for inference scoring: default=%default")
    parser.add_option("--seed", type=int, default=42,
                      help="Random seed: default=%default")
    parser.add_option("--no-4bit", action="store_true", default=False,
                      help="Disable 4-bit QLoRA (auto-set on MPS/CPU)")
    parser.add_option("--do-train", action="store_true", default=False,
                      help="Run fine-tuning")
    parser.add_option("--do-eval", action="store_true", default=False,
                      help="Run evaluation after training (or from --checkpoint)")

    (opts, _) = parser.parse_args()

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        opts.no_4bit = True
        logger.info("MPS detected: 4-bit quantization requires CUDA — disabling.")
    else:
        device = torch.device("cpu")
        opts.no_4bit = True

    logger.info(f"Device: {device}")

    # ── Output directory ─────────────────────────────────────────────────────
    model_short = opts.model.rstrip("/").split("/")[-1]
    outdir = os.path.join(
        opts.basedir, "llama",
        f"{opts.output_prefix}_{model_short}_s{opts.seed}_lr{opts.lr}",
    )
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"Output dir: {outdir}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    model_path = opts.checkpoint if (opts.checkpoint and not opts.do_train) else opts.model
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched next-token scoring

    label_token_ids = get_label_token_ids(tokenizer)

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info(f"Loading base model from {opts.model}")
    if not opts.no_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            opts.model,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        dtype = torch.float16 if device.type != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            opts.model,
            torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            model = model.to(device)

    # ── LoRA ─────────────────────────────────────────────────────────────────
    from peft import (
        LoraConfig, TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
    )

    if opts.checkpoint and not opts.do_train:
        # Load saved LoRA adapter weights for eval
        logger.info(f"Loading LoRA adapter from checkpoint: {opts.checkpoint}")
        model = PeftModel.from_pretrained(model, opts.checkpoint)
    else:
        if not opts.no_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=opts.lora_rank,
            lora_alpha=opts.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # ── Training ─────────────────────────────────────────────────────────────
    if opts.do_train:
        train_path = os.path.join(opts.basedir, opts.train_file)
        logger.info(f"Loading training data from {train_path}")
        train_items = load_jsonlist(train_path)
        logger.info(f"  {len(train_items)} examples")

        train_dataset = ToneDataset(
            train_items, tokenizer, max_length=opts.max_seq_length
        )

        training_args = TrainingArguments(
            output_dir=outdir,
            num_train_epochs=opts.n_epochs,
            per_device_train_batch_size=opts.batch_size,
            gradient_accumulation_steps=opts.grad_accum,
            learning_rate=opts.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            fp16=(device.type == "cuda"),
            logging_dir=os.path.join(outdir, "logs"),
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=1,
            seed=opts.seed,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
        )

        logger.info("=== Starting training ===")
        trainer.train()
        trainer.save_model(outdir)
        tokenizer.save_pretrained(outdir)
        logger.info(f"Saved to {outdir}")

    # ── Evaluation ───────────────────────────────────────────────────────────
    if opts.do_eval:
        for split_name, split_file in [
            ("dev", opts.dev_file),
            ("test", opts.test_file),
        ]:
            split_path = os.path.join(opts.basedir, split_file)
            if not os.path.exists(split_path):
                logger.warning(f"Not found, skipping: {split_path}")
                continue
            logger.info(f"Evaluating on {split_path}")
            items = load_jsonlist(split_path)
            true_idx, pred_idx, probs = score_examples(
                model, tokenizer, items, label_token_ids, device,
                batch_size=opts.eval_batch_size,
                max_length=opts.max_seq_length,
            )
            save_predictions(outdir, split_name, true_idx, pred_idx, probs)


if __name__ == "__main__":
    main()
