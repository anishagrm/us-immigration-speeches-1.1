"""
classification/run_llama_qlora_relevance.py

Fine-tune meta-llama/Llama-3.2-1B for immigration-speech relevance classification
using QLoRA (4-bit NF4 + LoRA rank-16/alpha-16 on q_proj, v_proj).

Binary task: is this speech segment about immigration? yes / no

At inference, the label is determined by comparing the log-probabilities the model
assigns to "yes" and "no" given the formatted input prompt.

Data labels:  yes | no   (sorted: no=0, yes=1)
Output TSV columns match hf/run.py: true, predicted, 0, 1

Usage:
  python -m classification.run_llama_qlora_relevance \\
      --basedir data/speeches/Congress/relevance/splits/basic/ \\
      --train-file all.jsonlist \\
      --dev-file dev.jsonlist \\
      --test-file test.jsonlist \\
      --output-prefix llama_qlora \\
      --do-train --do-eval \\
      --n-epochs 3 --seed 42
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
# Sorted alphabetically → matches hf/run.py index order
SORTED_LABELS = ["no", "yes"]   # indices 0, 1
LABEL2IDX = {l: i for i, l in enumerate(SORTED_LABELS)}
LABEL2TOKEN = {"no": "no", "yes": "yes"}

# ── Annotator guidelines ─────────────────────────────────────────────────────
ANNOTATOR_GUIDELINES = """\
You are an expert annotator of political speech. Your task is to classify \
whether a Congressional speech segment is relevant to the topic of immigration.

ANNOTATION GUIDELINES
---------------------
yes   The segment is about immigration, immigrants, or closely related topics
      such as border policy, citizenship, visas, asylum, deportation, or the
      rights and treatment of foreign-born people. The segment discusses people
      moving between countries for any reason (work, refuge, family, etc.).

no    The segment is not about immigration. It may mention foreign countries,
      international trade, foreign policy, or ethnic groups without discussing
      immigration specifically. Procedural remarks, unrelated legislation, or
      general political debate that does not concern immigration should be
      marked no.

Respond with exactly one word: yes or no.
"""


def build_prompt(text: str, label: str = None) -> str:
    prompt = (
        f"{ANNOTATOR_GUIDELINES}\n"
        f"Speech segment:\n\"{text}\"\n\n"
        "Relevant to immigration? "
    )
    if label is not None:
        prompt += LABEL2TOKEN[label]
    return prompt


def load_jsonlist(path: str):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class RelevanceDataset(Dataset):
    def __init__(self, items, tokenizer, max_length: int = 512):
        self.examples = []
        for item in items:
            text = item["text"]
            label = item["label"]

            full_prompt = build_prompt(text, label=label)
            prompt_only = build_prompt(text, label=None)

            full_enc = tokenizer(full_prompt, max_length=max_length, truncation=True)
            prompt_len = len(
                tokenizer(prompt_only, max_length=max_length, truncation=True)["input_ids"]
            )

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            self.examples.append(
                {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {k: torch.tensor(v) for k, v in ex.items()}


def collate_fn(batch, pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids_out, attn_out, labels_out = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids_out.append(
            torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), x["input_ids"]])
        )
        attn_out.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), x["attention_mask"]])
        )
        labels_out.append(
            torch.cat([torch.full((pad_len,), -100, dtype=torch.long), x["labels"]])
        )
    return {
        "input_ids": torch.stack(input_ids_out),
        "attention_mask": torch.stack(attn_out),
        "labels": torch.stack(labels_out),
    }


def get_label_token_ids(tokenizer) -> list:
    """Return token IDs for 'no' and 'yes' in SORTED_LABELS order."""
    ids = []
    for tok in [LABEL2TOKEN[l] for l in SORTED_LABELS]:
        for candidate in [tok, " " + tok]:
            enc = tokenizer.encode(candidate, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
                break
        else:
            enc = tokenizer.encode(tok, add_special_tokens=False)
            ids.append(enc[0])
            logger.warning(f"Token '{tok}' is multi-subword; using first subtoken {enc[0]}")
    logger.info(
        "Label token IDs  "
        + "  ".join(f"{l}={i}" for l, i in zip(SORTED_LABELS, ids))
    )
    return ids


@torch.no_grad()
def score_examples(model, tokenizer, items, label_token_ids, device,
                   batch_size: int = 8, max_length: int = 512):
    model.eval()
    true_indices, all_probs = [], []

    for start in tqdm(range(0, len(items), batch_size), desc="Scoring"):
        batch = items[start: start + batch_size]
        prompts = [build_prompt(item["text"]) for item in batch]

        enc = tokenizer(
            prompts, max_length=max_length, truncation=True,
            padding=True, return_tensors="pt",
        ).to(device)

        outputs = model(**enc)
        logits = outputs.logits
        seq_lens = enc["attention_mask"].sum(dim=1) - 1

        for i, item in enumerate(batch):
            last_pos = seq_lens[i].item()
            label_logits = logits[i, last_pos, label_token_ids]  # (2,)
            probs = torch.softmax(label_logits, dim=0).cpu().float().numpy()
            all_probs.append(probs)
            true_indices.append(LABEL2IDX.get(item.get("label", ""), -1))

    probs_matrix = np.array(all_probs)
    predicted = np.argmax(probs_matrix, axis=1)
    return np.array(true_indices), predicted, probs_matrix


def save_predictions(outdir, split_name, true_indices, predicted, probs_matrix):
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
    f1 = f1_score(t, p, pos_label=LABEL2IDX["yes"], zero_division=0)
    macro_f1 = f1_score(t, p, average="macro", zero_division=0)

    lines = [f"acc = {acc:.4f}", f"f1-yes = {f1:.4f}", f"macro_f1 = {macro_f1:.4f}"]
    result_str = "\n".join(lines)
    logger.info(f"\n=== {split_name} ===\n{result_str}")
    with open(os.path.join(outdir, f"eval_results_{split_name}.txt"), "w") as f:
        f.write(result_str + "\n")


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option("--model", type=str, default="meta-llama/Llama-3.2-1B",
                      help="HuggingFace model name or path: default=%default")
    parser.add_option("--checkpoint", type=str, default=None,
                      help="Path to a saved PEFT checkpoint for eval-only")
    parser.add_option("--basedir", type=str,
                      default="data/speeches/Congress/relevance/splits/basic/",
                      help="Base data directory: default=%default")
    parser.add_option("--train-file", type=str, default="all.jsonlist")
    parser.add_option("--dev-file", type=str, default="dev.jsonlist")
    parser.add_option("--test-file", type=str, default="test.jsonlist")
    parser.add_option("--output-prefix", type=str, default="llama_qlora")
    parser.add_option("--n-epochs", type=int, default=3)
    parser.add_option("--lr", type=float, default=2e-4)
    parser.add_option("--batch-size", type=int, default=4)
    parser.add_option("--grad-accum", type=int, default=4)
    parser.add_option("--max-seq-length", type=int, default=512)
    parser.add_option("--lora-rank", type=int, default=16)
    parser.add_option("--lora-alpha", type=int, default=16)
    parser.add_option("--eval-batch-size", type=int, default=8)
    parser.add_option("--seed", type=int, default=42)
    parser.add_option("--no-4bit", action="store_true", default=False)
    parser.add_option("--do-train", action="store_true", default=False)
    parser.add_option("--do-eval", action="store_true", default=False)

    (opts, _) = parser.parse_args()

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        opts.no_4bit = True
    else:
        device = torch.device("cpu")
        opts.no_4bit = True
    logger.info(f"Device: {device}")

    model_short = opts.model.rstrip("/").split("/")[-1]
    outdir = os.path.join(
        opts.basedir, "llama",
        f"{opts.output_prefix}_{model_short}_s{opts.seed}_lr{opts.lr}",
    )
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"Output dir: {outdir}")

    model_path = opts.checkpoint if (opts.checkpoint and not opts.do_train) else opts.model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    label_token_ids = get_label_token_ids(tokenizer)

    if not opts.no_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            opts.model, quantization_config=bnb_config, device_map="auto"
        )
    else:
        dtype = torch.float16 if device.type != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            opts.model, torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            model = model.to(device)

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel

    if opts.checkpoint and not opts.do_train:
        model = PeftModel.from_pretrained(model, opts.checkpoint)
    else:
        if not opts.no_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=opts.lora_rank, lora_alpha=opts.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    if opts.do_train:
        train_path = os.path.join(opts.basedir, opts.train_file)
        train_items = load_jsonlist(train_path)
        logger.info(f"Loaded {len(train_items)} training examples from {train_path}")

        train_dataset = RelevanceDataset(train_items, tokenizer, max_length=opts.max_seq_length)

        training_args = TrainingArguments(
            output_dir=outdir,
            num_train_epochs=opts.n_epochs,
            per_device_train_batch_size=opts.batch_size,
            gradient_accumulation_steps=opts.grad_accum,
            learning_rate=opts.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            fp16=(device.type == "cuda"),
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

    if opts.do_eval:
        for split_name, split_file in [("dev", opts.dev_file), ("test", opts.test_file)]:
            split_path = os.path.join(opts.basedir, split_file)
            if not os.path.exists(split_path):
                logger.warning(f"Not found, skipping: {split_path}")
                continue
            items = load_jsonlist(split_path)
            true_idx, pred_idx, probs = score_examples(
                model, tokenizer, items, label_token_ids, device,
                batch_size=opts.eval_batch_size, max_length=opts.max_seq_length,
            )
            save_predictions(outdir, split_name, true_idx, pred_idx, probs)


if __name__ == "__main__":
    main()
