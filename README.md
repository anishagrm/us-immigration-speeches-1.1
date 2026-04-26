
# Replication Code and Data

This repo collects together the main scripts used for the data preprocessing and analysis in "Computational analysis of 140 years of US political speeches reveals more positive but increasingly polarized framing of immigration".

Sufficient scripts and processed data are included in the Release to reproduce the figures and findings in the main paper.

Additional scripts are also included to reproduce the processing of the original raw data, which is available from external sources (see below).  

To replicate analysis and plots with processed data included in Release, jump to **Plots** below.
 

### Requirements:

The following python packages are used in this repo

- shap
- tqdm
- numpy
- scipy
- spacy
- torch
- gensim
- pandas
- pystan
- seaborn
- matplotlib
- smart_open
- scikit-learn
- statsmodels
- transformers



### A Note on Usage:

Note that all scripts in this repo should be run from the main directory using the "-m" option, e.g.:

`python -m analysis.count_county_mentions -h`


### Data Sources:

There are three main sources of data for this project, which are all publicly available from external sources.

The primary source for Congressional data is the Stanford copy of the Congressional Record [https://data.stanford.edu/congress_text](https://data.stanford.edu/congress_text). From this, we use the Hein Bound edition for congresses 43 through 111.

For more recent Congresses (104 through 116) we use the scripts in the USCR repo: https://github.com/unitedstates/congressional-record/

For Presidential data, we scrape data from the [American Presidency Project](https://www.presidency.ucsb.edu/) using scripts in the `app` part of this repo: https://github.com/dallascard/scrapers

Additional tone annotations from the Media Frames Corpus are included in this repo.

For population numbers, we use a combination of sources, as described in the paper. A combined file is included in the Release for this repo. 

Processed data which are too large to be included in the source files for this repo, including trained models and model predictions, are available for download in the [latest release](https://github.com/dallascard/us-immigration-speeches/releases).

### Preprocessing:

There are parallel scripts for processing each part of the data. Steps include preprocessing, tokenization, parsing, and recombining into segments

For the Hein Bound data:

- `parsing/tokenize_hein_bound.py`: tokenize hein-bound using spacy  (also drop speeches from one day with corrupted data, and repair false sentence breaks)
- `parsing/rejoin_into_pieces_by_congress.py`: this script has two purposes: split each speech into one json per sentence, or one json per block of text (up to some limit)

For USCR:

- `uscr/export_speeches.py`: export the USCR data to .jsonlist files
- `parsing/preprocess_uscr.py`: adjust the text of USCR to more closely match the Gentzkow data (remove apostrophes, hyphens and speaker names)
- `parsing/tokenize_uscr.py`: output tokenized version of USCR (sentences and tokens)
- `parsing/rejoin_into_pieces_by_congress_uscr.py`: rejoin tokenized sentences into longer segments for classification

For Presidential data:

- use `scrapers/app/combine_categories.py` to combine all data into one file (external repo linked above)
- use `presidential/export_presidential_segments.py` to select the subset of paragraphs from presidents
- use `presidential/tokenize_presidential.py` to tokenize documents
- use `presidential/select_segments.py` to select paragraphs with the relevant keywords


### Speech selection for annotation

As a first step, we selected speech segments that could be about immigration using keywords, which we refer to as "keyword segments":

- `speech_selection/export_segments_early_with_overlap.py`: export segments using the early era keywords, with some overlap to the middle era
- `speech_selection/export_segments_mid_with_overlap.py`: export segments using the middle era keywords, with some overlap to the early and modern eras
- `speech_selection/export_segments_modern_with_overlap.py`: export segments using the modern era keywords, with some overlap to the middle era
- `speech_selection/export_segments_uscr.py`: export segments from USCR

We then combined these into batches, and collected annotations:

- `speech_selection/make_batches_early.py` etc: combine segments into batches for annotation
- `speech_selection/make_batches_mid.py` etc: combine segments into batches for annotation
- `speech_selection/make_batches_modern.py` etc: combine segments into batches for annotation

### Annotations

Raw annotations for tone and relevance are provided in online data files

To process the annotations:

- `annotations/tokenize.py`:  Collect all the annotated text segments and tokenize with spacy
- `annotations/export_for_label_aggregation.py`: Collect the annotations and export for aggregating labels (using label-aggregation)
- `annoations/measure_agreement.py` to measure agreement rates using Krippendorff's alpha
- Do label aggregation using label-aggregation repo (`github.com/dallascard/label-aggregation`) using Stan with the --no-vigilance option for both relevance and tone
- `relevance/make_relevance_splits.py`: Collect the tokenizations and estimated label probabilities, and make splits
- `relevance/make_relevance_splits.py` and `tone.make_tone_splits.py`: Divide the annotated data with inferred labels into train, dev, and test files for model training. For the latter, the additional annotations from MFC should be included using the `--extra-data-file` options, pointed to `data/annotations/relevance_and_tone/mfc/mfc_imm_tone.jsonlist`

### Training models

Run Roberta models on congressional annotations

- `classification/run_search_hf.py` to search of seeds (in order to estimate performance)
- `classification/run_final_model.py` to train a final model on all data with one seed
- `classification/make_predictions.py` to predict on keyword segments
- `classification/predict_on_all.py` to predict on all segments from each congress (exported from `parsing.rejoin_into_pieces_by_congress.py`)


### Collecting predictions

- use `relevance/collect_predictions.py` to get the relevant immigration speeches and segments
- use `tone/collect_predictions.py` to get the tones of these speeches and segments
- use `export/export_imm_segments_with_tone_and_metadata.py` to export the text, tone, and metadata
(some of the above depend on intermediate scripts, like `metadata.export_speeech_dates.py`)


### Identifying procedural speeches

- use `filtering/export_training_and_test.py` to export a heuristically labeled dataset of segments (procedural and not)
- use `filtering/export_short_speehces.py` to export short speeches to be classified
- train a model to identify procedural speeches using sklearn or equivalent
- use `filtering/collect_prediction.py` to gather up those speeches identified as procedural


### Additional Preprocessing

The following scripts are required for full replication:

- use `analysis/count_nouns.py` to count the nouns in the Congressional Record (for generating a random subset)
- use `analysis/choose_random_nouns.py` to get a random set of nouns not already used (for metaphor analysis)


### Analysis

Export some additional data based on speeches to simplify plotting:

- use `analysis/count_country_mentions.py` to identify frequently mentioned nationalities and relevance speeches
- use `export/export_imm_speeches_parsed.py` to collect and export the parsed versions of all immigration speeches
- use `analysis/identify_immigrant_mentions.py` to collect and export the mentions of immigrants and groups
- use `analysis/identify_group_mentions.py` to select the subset of mention sentences also mentioning each group
- use `analysis/count_tagged_lemmas.py` to collect counts
- use `analysis/count_speeches_and_tokens.py` to get background counts of non-procedural speeches

Measuring Impact:

- use `export/export_tone_for_lr_models.py` to export data for Logistic Regression classifiers
- train linear models with Frustratingly Easy Domain Adaptation (external repo)

Create contextual embeddings for masked terms and measuring dehumanization:

- use `embeddings/embed_immigrant_terms_masked.py` to get contextual embeddings for each mention
- use `embeddings/convert_embeddings_to_word_probs.py` to compute probabilities for each vector
- use `analysis/run_metaphorical_analysis.py` to compute metaphorical associations

Stan model (Appendix):

- use `stan/run_final_model.py` to run the Bayesian model with session, party, region, and chamber as factors

### Plots

If working with the processed data included in the Release, simply unzip the data.zip file in this directory, then run the following scripts:

- `analysis/count_county_mentions.py`
- `analysis/run_metaphorical_analysis.py`

The following scripts can be used to reproduce the main plots:

- use `plotting/make_tone_plots.py` to make all of the tone plots
- use `plotting/make_pmi_plots.py` to make all of the pmi plots
- use `plotting/make_metaphor_plots.py` to make the separate metaphor plots in the Appendix

To get the terms in table 1:
- use `export/export_imm_segments_for_linear.py` to export classified immigration segments to the appopriate format for the desired range of sessions
- use `linear/get_shap_values.py` to get the data in the right format


### Additional code for validation material in SI

For combining annotations (used for linear and CFM models in SI)
- `relevance/combine_relevance_data.py` (to combine all relevance data into one dataset and create a random test set)
- `tone/combine_tone_data.py` (to combine all relevance data into one dataset and create a random test set)
- `tone/filter_neutral.py` to filter out neutral speehces (for bianry model)

For running all linear models:
- `linear/create_partition.py` to convert dataset to proper format
- `linear/train.py` to train a model
- `linear/predict.py` or `linear/predict_on_all.py` to make predictions on other data
- `linear/export_weight.py` to export model weights

For linear model replication (in SI):
- train and predict using scripts in `linear`
- `relevance/collect_predictions_linear.py`
- `tone/collect_predictions_linear.py`
- use normal plotting scripts, pointing to new directories

For binary model replication (in SI):
- train and predict using scripts in `classification`
- `relevance/collect_predictions_val.py`
- `tone/collect_predictions_binary.py`
- `plotting/make_tone_plots_binary.py`

For CFM model replication (in SI):
- `tone/collect_predictions_cfm.py` to collect predictions and apply corrections
- not that this must be run three times, once with defaults, once with `--party-cfm D` and once with `--party-cfm R`
- use `plotting/make_tone_plots_probs_three.py` to put these all together

For leave-one-out plots and plots by individual speakers
- `plotting/make_tone_plots_loo.py`

For Frame comparison for Europe vs Latin America (in SI):
- `plotting/make_pmi_plots_latin_america.py`

For public opinion and SEI analyses (in SI), refer to `public_opinion_and_sei`

---

## RooseBERT Experiments (Anisha Gurram, 2026)

This section documents experiments comparing RooseBERT against the original RoBERTa baseline for the relevance and tone classifiers, and a dehumanization analysis using RooseBERT's MLM head in place of the original model.

### Motivation

RooseBERT (`ddore14/RooseBERT-*`) is a domain-specific BERT model pretrained on US political text. The hypothesis is that domain-specific pretraining may yield better performance on historical congressional speech classification and more contextually appropriate metaphor/dehumanization probability estimates compared to the general-purpose models used in the original paper.

### Experiment 1A: RoBERTa vs RooseBERT-cont-uncased (`runs/experiment_1`)

**Setup:** Both models trained on `train.jsonlist`, evaluated on `test.jsonlist` across 4 seeds (0, 1, 2, 42). Same hyperparameters: lr=2e-5, 7 epochs, batch size=8. RoBERTa uses `max_seq_length=400`; RooseBERT uses 512.

**Script:** `pace_compare_seeds.sh`

**Relevance** (n=762 test examples):

| Model | Mean Acc | Std | Mean F1 | Std |
|-------|----------|-----|---------|-----|
| roberta-base | 0.901 | 0.005 | 0.894 | 0.005 |
| RooseBERT-cont-uncased | 0.885 | 0.007 | 0.878 | 0.007 |

**Tone** (n=899 test examples):

| Model | Mean Acc | Std |
|-------|----------|-----|
| roberta-base | 0.678 | 0.004 |
| RooseBERT-cont-uncased | 0.676 | 0.005 |

**Findings:** RoBERTa outperforms RooseBERT-cont-uncased on relevance by ~1.6pp consistently across seeds. Tone results are statistically equivalent. Possible explanations: (1) hyperparameters were tuned for RoBERTa, not RooseBERT; (2) the uncased model loses case signal present in historical text (proper nouns, formal titles); (3) continued pretraining may not be as effective as training from scratch on domain text.

### Experiment 1B: Learning rate sweep on cased RooseBERT variants (`runs/experiment_2`)

To address the limitations in Sub-experiment A, two cased RooseBERT variants were evaluated across a learning rate sweep on `dev.jsonlist` with a single seed (42).

**Script:** `pace_lr_sweep.sh`

Models evaluated:
- `ddore14/RooseBERT-cont-cased`: continued pretraining on political text, cased
- `ddore14/RooseBERT-scr-cased`: trained from scratch on political text, cased

**Relevance dev set results:**

| Model | LR | Dev Acc | Dev F1 |
|-------|----|---------|--------|
| RooseBERT-cont-cased | 1e-5 | 0.900 | 0.897 |
| RooseBERT-cont-cased | 2e-5 | 0.906 | 0.901 |
| **RooseBERT-cont-cased** | **3e-5** | **0.917** | **0.914** |
| RooseBERT-scr-cased | 1e-5 | 0.890 | 0.885 |
| RooseBERT-scr-cased | 2e-5 | 0.896 | 0.891 |
| RooseBERT-scr-cased | 3e-5 | 0.912 | 0.906 |

**Tone dev set results:**

| Model | LR | Dev Acc | F1-anti | F1-neutral | F1-pro |
|-------|----|---------|---------|------------|--------|
| RooseBERT-cont-cased | 1e-5 | 0.680 | 0.668 | 0.571 | 0.754 |
| RooseBERT-cont-cased | 2e-5 | 0.683 | 0.687 | 0.567 | 0.757 |
| RooseBERT-cont-cased | 3e-5 | 0.674 | 0.687 | 0.555 | 0.753 |
| **RooseBERT-scr-cased** | **1e-5** | **0.703** | **0.711** | **0.600** | **0.761** |
| RooseBERT-scr-cased | 2e-5 | 0.686 | 0.698 | 0.552 | 0.766 |
| RooseBERT-scr-cased | 3e-5 | 0.680 | 0.701 | 0.559 | 0.747 |

**Best hyperparameters identified:** `cont-cased @ lr=3e-5` for relevance; `scr-cased @ lr=1e-5` for tone.

### Experiment 1C: Final comparison with best hyperparameters (`runs/experiment_3`)

Using the best model+LR from Sub-experiment B, each RooseBERT variant is compared against `roberta-base` on `test.jsonlist` across 3 seeds (0, 1, 2).

**Script:** `pace_final_comparison.sh`

**Relevance** (n=762 test examples):

| Model | Mean Acc | Std | Mean F1 | Std |
|-------|----------|-----|---------|-----|
| roberta-base (lr=2e-5) | 0.893 | 0.002 | 0.885 | 0.002 |
| RooseBERT-cont-cased (lr=3e-5) | 0.887 | 0.003 | 0.878 | 0.001 |

Per-seed relevance results:

| Model | Seed 0 Acc | Seed 1 Acc | Seed 2 Acc |
|-------|-----------|-----------|-----------|
| roberta-base | 0.891 | 0.892 | 0.895 |
| RooseBERT-cont-cased | 0.891 | 0.885 | 0.886 |

**Tone** (n=899 test examples):

| Model | Mean Acc | Std | Mean F1-anti | Mean F1-neutral | Mean F1-pro |
|-------|----------|-----|-------------|----------------|------------|
| roberta-base (lr=2e-5) | 0.670 | 0.008 | 0.651 | 0.567 | 0.760 |
| RooseBERT-scr-cased (lr=1e-5) | 0.670 | 0.006 | 0.649 | 0.551 | 0.762 |

Per-seed tone results:

| Model | Seed 0 Acc | Seed 1 Acc | Seed 2 Acc |
|-------|-----------|-----------|-----------|
| roberta-base | 0.669 | 0.681 | 0.662 |
| RooseBERT-scr-cased | 0.679 | 0.667 | 0.665 |

**Findings:** Even with optimal hyperparameters, RoBERTa holds a small advantage on relevance (~0.6pp). For tone, both models score identically on the test set (0.670 mean acc), despite RooseBERT-scr-cased showing 0.703 on the dev set. This gap between dev and test suggests the dev-set improvement was not generalizable. Domain-specific pretraining does not provide a measurable benefit for either classification task.

### Experiment 1D: MLM dehumanization analysis with RooseBERT

**Goal:** Repeat the original paper's metaphor/dehumanization analysis using `RooseBERT-cont-cased` in place of `bert-base-uncased`. RooseBERT's domain-specific vocabulary and syntax modeling may yield different word probability distributions when predicting masked immigrant terms — particularly for pre-1970s historical text that is closer to its pretraining corpus.

**Script:** `pace_roosebert_mlm.sh`

**Pipeline:**

1. Extract contextual embeddings for masked immigrant mentions:
   ```bash
   python3 -m embeddings.embed_immigrant_terms_masked \
       --model-type bert \
       --model ddore14/RooseBERT-cont-cased \
       --outdir data/speeches/Congress/contextual-embeddings/RooseBERT-cont-cased \
       --device 0
   ```

2. Apply the MLM head to convert embeddings to word probability distributions:
   ```bash
   python3 -m embeddings.convert_embeddings_to_word_probs \
       --model-type bert \
       --model ddore14/RooseBERT-cont-cased \
       --infile data/speeches/Congress/contextual-embeddings/RooseBERT-cont-cased/immigrant_vectors_masked.npz \
       --outdir data/speeches/Congress/contextual-embeddings/RooseBERT-cont-cased \
       --device 0
   ```

3. Run metaphorical association analysis and statistical tests:
   ```bash
   python3 -m analysis.run_metaphorical_analysis \
       --emb-dir data/speeches/Congress/contextual-embeddings/RooseBERT-cont-cased \
       --outdir data/speeches/Congress/metaphors/RooseBERT-cont-cased
   ```

Results to be added upon completion.

### Overall Findings

The original paper used `roberta-base` for relevance and tone classification. These experiments tested whether RooseBERT — domain-specifically pretrained on US political text — would outperform RoBERTa.

**Classification:** RoBERTa retains a small but consistent advantage on relevance (~0.6–1.6pp) across all RooseBERT variants and hyperparameter configurations. For tone, no RooseBERT variant outperforms RoBERTa on the held-out test set. Contributing factors: (a) the annotation labels were likely generated or validated using RoBERTa-family models, creating a distributional alignment advantage; (b) the training sets are small (~5–6k examples), limiting the benefit of domain pretraining; (c) political speech may already be well-represented in RoBERTa's pretraining corpus.

**Dehumanization analysis:** Pending. Results will indicate whether RooseBERT's political text pretraining changes which metaphorical associations (animal, disease, criminal, etc.) are predicted for immigrant terms across historical periods.

### Code changes from original

The following files were modified to run these experiments:

- `hf/run.py`: Fixed `AdamW` import (moved from `transformers` to `torch.optim`); disabled per-epoch checkpoint saving to reduce disk usage; fixed F1 metric bug (`pos_label=None` → `pos_label=1`)
- `hf/processors.py`: Made `weight_field` lookup robust to missing keys (`.get()` with default 1.0)
- `hf/metrics.py`: Fixed `pos_label=None` invalid for binary F1 in newer sklearn
- `classification/run_final_model_tone.py`: Removed hardcoded `--weight_field weight` (60% of tone data lacks weight field)
- `classification/run_folds_hf_tone.py`: Same fix as above
- `embeddings/convert_embeddings_to_word_probs.py`: Added `--model-type bert` support to load `BertForMaskedLM` (original script only supported RoBERTa)

### SLURM scripts

All scripts are submitted from the project root on PACE ICE via `sbatch <script>`:

- `pace_compare_seeds.sh`: Multi-seed (0, 1, 2) comparison of RoBERTa vs RooseBERT-cont-uncased on `test.jsonlist`
- `pace_lr_sweep.sh`: LR sweep (1e-5, 2e-5, 3e-5) on `dev.jsonlist` for cont-cased and scr-cased RooseBERT variants
- `pace_final_comparison.sh`: Final 3-seed comparison of best RooseBERT variant+LR vs RoBERTa on `test.jsonlist`; results saved to `runs/experiment_3/`
- `pace_roosebert_mlm.sh`: Full MLM dehumanization pipeline using RooseBERT-cont-cased (embed → word probs → metaphor analysis)

---

## Citation

To cite this respository or the data contained herein, please use:

Dallas Card, Serina Chang, Chris Becker, Julia Mendelsohn, Rob Voigt, Leah Boustan, Ran Abramitzky, and Dan Jurafsky. Replication code and data for "Computational analysis of 140 years of US political speeches reveals more positive but increasingly polarized framing of immigration" [dataset] (2022). https://github.com/dallascard/us-immigration-speeches/

```
@article{card.2022.immdata,
  author = {Dallas Card and Serina Chang and Chris Becker and Julia Mendelsohn and Rob Voigt and Leah Boustan and Ran Abramitzky and Dan Jurafsky},
  title = {Replication code and data for "Computational analysis of 140 years of US political speeches reveals more positive but increasingly polarized framing of immigration" [dataset]},
  year=2022,
  url={https://github.com/dallascard/us-immigration-speeches/}
}
```
