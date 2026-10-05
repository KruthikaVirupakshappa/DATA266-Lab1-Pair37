# DATA266 Lab 1 — Pair 37

LLM pretraining (Task 1) · Sentiment classification (Task 2) · CycleGAN style transfer (Task 3).

## Setup

Tested with Python 3.13 on macOS (Apple M3, MPS). Also runs on CUDA or CPU; the device is chosen automatically.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test (one command)

Run from the repo root:

```bash
python task1_llm/kruthika/src/smoke_test.py
```

If the dataset isn't there yet, this first downloads it (see [Data](#data)). It then runs a short end-to-end pipeline on a 2,000-story subset: data → model → ~200 training steps → evaluation → checkpoint save/reload → text generation. The whole run takes about a minute. It checks:

| Check | What it verifies |
|---|---|
| data | split sizes, `char_to_idx`/`idx_to_char` round trip, next-character targets |
| model | output shapes, finite loss, **causal masking** (changing token *t* must not change logits before *t*) |
| train | loss decreases, no NaN/inf losses or gradients |
| eval | val loss, perplexity, bits-per-char, top-1 accuracy |
| checkpoint | reloaded weights give identical logits |
| generate | continuation longer than the context window |

Exit code `0` means every check passed. Each run writes a raw log to `reproducibility/raw_logs/` and a manifest (package versions, git commit, config, metrics) to `reproducibility/manifests/`.

## Task 1 full training (kruthika)

Trained on a Google Colab GPU with [task1_colab.ipynb](task1_llm/kruthika/src/task1_colab.ipynb). That notebook downloads the dataset and then runs:

```bash
python task1_llm/kruthika/src/train.py
```

Add `--epochs N` to change the number of epochs (default 10, set in `src/config.yaml`) and `--resume` to continue after a disconnect. Results go to `task1_llm/kruthika/metrics_report.csv` and `task1_llm/kruthika/outputs/<run_name>/` (loss curves, samples, histories). Best weights go to `task1_llm/kruthika/checkpoints/<run_name>_best.pt`.

**Results** (run `gpt_L6_H6_D384`: 6 layers, 6 heads, 384 dims, 10.8M parameters, 10 epochs): validation loss 0.536, perplexity 1.71, 0.773 bits-per-character, 82.8% next-character accuracy. Details:
- [results.md](task1_llm/kruthika/results.md): architecture, hyperparameters and all metrics
- [failure_analysis.md](task1_llm/kruthika/failure_analysis.md): three failure cases
- [metrics_report.csv](task1_llm/kruthika/metrics_report.csv)
- raw log and manifest in `reproducibility/`

### Task 1 team comparison (shared test set)

Both members' models are compared on a shared test set of 5,000 official-validation stories that neither model has seen:

```bash
python task1_llm/shared_eval/build_test_set.py    # writes task1_llm/shared_eval/test_set.json
python task1_llm/shared_eval/compare_models.py    # needs both members' folders (task1_llm/kruthika, task1_llm/jiayi)
```

Each model uses its own preprocessing, vocabulary and context length; generation uses identical prompts and settings. Results: [comparison_table.md](task1_llm/shared_eval/comparison_table.md) and [comparison_report.csv](task1_llm/shared_eval/comparison_report.csv).

## Task 2 sentiment classification (kruthika)

Trained on a Google Colab GPU with [task2_colab.ipynb](task2_sentiment/kruthika/src/task2_colab.ipynb). That notebook downloads the dataset and then runs:

```bash
python task2_sentiment/kruthika/src/train.py
```

This preprocesses the data, then trains and evaluates all three models. Add `--epochs N` to change the maximum epochs per model (default 5, with early stopping). Models that already finished are not retrained; add `--force` to retrain them.

**Results** (official Yelp Polarity test set, 38,000 reviews; embeddings learned from scratch):

| Model | Accuracy [95% CI] | Macro-F1 | MCC | ROC-AUC |
|---|---|---|---|---|
| Baseline: mean embedding | 0.9326 [0.9301, 0.9352] | 0.9326 | 0.8652 | 0.9790 |
| Exp 1: BiLSTM | **0.9531** [0.9509, 0.9550] | **0.9531** | **0.9062** | **0.9902** |
| Exp 2: TextCNN | 0.9435 [0.9412, 0.9456] | 0.9435 | 0.8871 | 0.9867 |

Details:
- [results.md](task2_sentiment/kruthika/results.md): preprocessing, models, all metrics, McNemar tests, slices
- [failure_analysis.md](task2_sentiment/kruthika/failure_analysis.md): review of 20 errors
- [metrics_report.csv](task2_sentiment/kruthika/metrics_report.csv)
- raw log and manifest in `reproducibility/`

## Task 3 CycleGAN Monet ↔ Photo (kruthika)

Trained on a Google Colab GPU with [task3_colab.ipynb](task3_gan/kruthika/src/task3_colab.ipynb). That notebook downloads the competition data (needs a Kaggle API token in Colab Secrets) and runs:

```bash
python task3_gan/kruthika/src/train.py --config task3_gan/kruthika/src/config.yaml      # v1: 60 epochs from scratch
python task3_gan/kruthika/src/train.py --config task3_gan/kruthika/src/config_v2.yaml   # v2: fine-tune of v1 (final model)
python task3_gan/kruthika/evaluate_local.py --config task3_gan/kruthika/src/config_v2.yaml
```

`evaluate_local.py` writes `pred_A2B/` and `pred_B2A/`, `submission.csv` (computed exactly as the instructor's evaluation script: first 300 files per folder, average FID and MiFID over both directions), `full_metrics_report.csv` and the human-audit sheets. Training resumes automatically from the last checkpoint after a disconnect.

**Results** (final model v2): Kaggle **−48.3416** (rank 19 on 2026-10-05); FID photo→Monet 97.44, Monet→photo 95.10 (submitted average 96.27, MiFID 0.4115). Details:
- [results.md](task3_gan/kruthika/results.md): architecture, training runs, convergence and stability, all metrics
- [failure_analysis.md](task3_gan/kruthika/failure_analysis.md): visual quality and failure cases
- [full_metrics_report.csv](task3_gan/kruthika/full_metrics_report.csv), [submission.csv](task3_gan/kruthika/submission.csv)
- raw logs and manifests in `reproducibility/`

## Data

Raw data is not committed. To fetch it for **Task 1:**

```bash
python task1_llm/scripts/download_tinystories.py
```

This saves [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) at a pinned revision to `task1_llm/data/tinystories/` (~1.8 GB, 2.1M train and 22K validation stories).

**Train/validation split (Task 1).** Each member samples their own split with a seed from their config:
- **Validation (10K):** sampled from the official `validation` split.
- **Training (100K):** sampled from the official `train` split.
- The `train` split contains 6,601 exact copies of `validation` stories. Any training story identical to a chosen validation story is skipped, so validation text is never trained on.

**Task 2:**

```bash
python task2_sentiment/scripts/download_yelp_polarity.py
```

This saves [fancyzhx/yelp_polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity) at a pinned revision to `task2_sentiment/data/yelp_polarity/` (560K train and 38K test reviews, perfectly balanced).

**Task 3** (needs a Kaggle API token and having joined the competition):

```bash
python task3_gan/scripts/download_kaggle_data.py
```

This saves the class competition data to `task3_gan/data/`: `monet_jpg/` (300 paintings) and `photo_jpg/` (7,038 photos).

## Layout

```
task1_llm/
  data/tinystories/     shared raw TinyStories (git-ignored)
  scripts/              download_tinystories.py
  <member>/
    src/                code, config.yaml, notebook
    data_processed/     own preprocessing output (vocab, split stats)
    checkpoints/        trained weights
    outputs/            samples, plots, metrics
    metrics_report.csv  failure_analysis.md  results.md
task2_sentiment/  task3_gan/     same per-member structure
reproducibility/
  raw_logs/             unedited logs from every run
  manifests/            one JSON manifest per run
report/                 DATA266_Lab1_Report_Team_37.pdf
```

All paths in configs are relative to the repo root; there are no personal paths or secrets.
