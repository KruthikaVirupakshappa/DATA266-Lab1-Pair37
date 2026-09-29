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

Trained on a Google Colab GPU with [task1_colab.ipynb](task1_llm/kruthika/src/task1_colab.ipynb). That notebook runs the smoke test and then:

```bash
python task1_llm/kruthika/src/train.py
```

Add `--resume` to continue after a disconnect. Results go to `task1_llm/kruthika/metrics_report.csv` and `task1_llm/kruthika/outputs/<run_name>/` (loss curves, samples, histories). Best weights go to `task1_llm/kruthika/checkpoints/<run_name>_best.pt`.

## Data

Raw data is not committed. To fetch it:

```bash
python task1_llm/scripts/download_tinystories.py
```

This saves [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) at a pinned revision to `task1_llm/data/tinystories/` (~1.8 GB, 2.1M train and 22K validation stories).

**Train/validation split (Task 1).** Each member samples their own split with a seed from their config:
- **Validation (10K):** sampled from the official `validation` split.
- **Training (100K):** sampled from the official `train` split.
- The `train` split contains 6,601 exact copies of `validation` stories. Any training story identical to a chosen validation story is skipped, so validation text is never trained on.

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
