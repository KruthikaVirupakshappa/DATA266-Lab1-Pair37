# Task 2 Yelp Polarity Sentiment Classification Results

## Overview

Three sentiment-classification models were trained on the Yelp Polarity dataset without pretrained embeddings or pretrained language models. All three models used word embeddings learned from scratch and were evaluated on the untouched official Yelp test split of 38,000 reviews.

The model lineup was designed as a progression from a simple pooled-embedding baseline to two experimental architectures:

1. **Baseline Mean+Max**: learned token embeddings with masked mean pooling and masked max pooling, followed by a linear classifier.
2. **Experiment 1 MLP**: learned token embeddings with masked mean pooling, followed by a two-hidden-layer MLP with dropout.
3. **Experiment 2 Attention**: learned token and positional embeddings with single-head scaled dot-product self-attention, masked mean pooling, and a linear classifier.

Experiment 1 achieved the strongest overall test metrics numerically, although its improvement over the baseline was not statistically significant under the paired McNemar test.

## Data preprocessing

The raw Yelp Polarity dataset contained 560,000 training reviews and 38,000 official test reviews. No invalid or malformed rows were removed.

| Split | Examples | Negative | Positive |
|---|---:|---:|---:|
| Original train | 560,000 | 280,000 | 280,000 |
| Training split | 504,000 | 252,000 | 252,000 |
| Validation split | 56,000 | 28,000 | 28,000 |
| Official test | 38,000 | 19,000 | 19,000 |

The original training split was divided into 90% training and 10% validation using stratification and random seed `4253`. The official Yelp test split was kept untouched for final evaluation.

Preprocessing used:

- lowercasing
- punctuation and special-character removal
- stopword removal
- Porter stemming
- explicit preservation of common negation meaning before punctuation removal
- a word-level vocabulary learned only from the training split
- maximum vocabulary size of 30,000 with minimum frequency 2
- `<PAD>` and `<UNK>` tokens
- padding/truncation to a fixed sequence length

The official training runs used a vocabulary size of 30,000 and sequence length of **185 tokens**, determined from the configured 95th-percentile rule with a maximum-length cap of 256.

Raw review-length statistics before preprocessing were:

| Statistic | Words |
|---|---:|
| Median (p50) | 97 |
| p90 | 282 |
| p95 | 372 |
| p99 | 608 |
| Mean | 133.03 |
| Maximum | 1,052 |

## Model architectures and rationale

### Baseline: Mean+Max embedding classifier

Architecture:

`Embedding(30000, 128) -> Masked Mean Pooling + Masked Max Pooling -> Concatenate -> Linear(256, 1) -> Sigmoid`

The baseline uses a simple learned word representation and pooling-based classifier. Mean pooling captures the overall sentiment signal across a review, while max pooling can retain strong token-level evidence. This provides a relatively small, interpretable reference model before adding deeper nonlinear layers or attention.

Total parameters: **3,840,257**

### Experiment 1: Mean-embedding MLP

Architecture:

`Embedding(30000, 128) -> Masked Mean Pooling -> Linear(128, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 32) -> ReLU -> Dropout(0.3) -> Linear(32, 1) -> Sigmoid`

This experiment keeps the same from-scratch embedding size while replacing the baseline's direct linear decision layer with a nonlinear MLP. The two hidden layers test whether additional nonlinear capacity improves sentiment separation, while dropout provides regularization.

Total parameters: **3,850,369**

### Experiment 2: Single-head self-attention classifier

Architecture:

`Token Embedding(30000, 128) + Learned Positional Embedding -> Q/K/V projections -> Single-Head Self-Attention(64) -> Output Projection -> Masked Mean Pooling -> Linear(64, 1) -> Sigmoid`

This experiment tests whether explicit token-to-token interactions and learned position information improve classification beyond pooled embeddings. It uses single-head scaled dot-product self-attention with padding masking and no causal mask because sentiment classification can use the full review context.

Total parameters: **3,892,417**

## Training configuration

| Setting | Baseline | Experiment 1 MLP | Experiment 2 Attention |
|---|---:|---:|---:|
| Embedding dimension | 128 | 128 | 128 |
| Pooling | Mean + Max | Mean | Mean after attention |
| Hidden dimensions | — | 64, 32 | — |
| Dropout | — | 0.3 | — |
| Attention dimension | — | — | 64 |
| Batch size | 64 | 64 | 64 |
| Learning rate | 0.001 | 0.001 | 0.001 |
| Epochs | 5 | 5 | 5 |
| Optimizer | Adam | Adam | Adam |
| Loss | Binary cross-entropy | Binary cross-entropy | Binary cross-entropy |
| Decision threshold | 0.5 | 0.5 | 0.5 |
| Random seed | 4253 | 4253 | 4253 |

The best checkpoint for each model was selected by validation loss and then used for test-set inference.

## Training behavior

All three models showed decreasing training loss throughout the five epochs. Validation loss reached its minimum before the final epoch for every model, then increased while training loss continued to decrease, indicating mild overfitting late in training.

| Model | Epoch 1 Train | Epoch 1 Val | Best Epoch | Best Val | Epoch 5 Train | Epoch 5 Val |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Mean+Max | 0.2424 | 0.2070 | 4 | 0.1952 | 0.1592 | 0.2017 |
| Experiment 1 MLP | 0.2325 | 0.1950 | 4 | 0.1899 | 0.1628 | 0.1985 |
| Experiment 2 Attention | 0.2266 | 0.1949 | 3 | 0.1937 | 0.1556 | 0.2022 |

Loss-curve evidence:

- `task2_sentiment/jiayi/outputs/baseline_meanmax_loss_curve_20261004_234803.png`
- `task2_sentiment/jiayi/outputs/experiment1_mlp_loss_curve_20261002_211627.png`
- `task2_sentiment/jiayi/outputs/experiment2_attention_loss_curve_20261003_135109.png`

## Evaluation

### Overall classification metrics

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Micro F1 | Weighted F1 | ROC-AUC | PR-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Mean+Max | 0.9272 | 0.9272 | 0.9272 | 0.9272 | 0.9272 | 0.9272 | 0.9789 | 0.9789 | 0.8543 |
| Experiment 1 MLP | 0.9283 | 0.9285 | 0.9283 | 0.9283 | 0.9283 | 0.9283 | 0.9796 | 0.9797 | 0.8568 |
| Experiment 2 Attention | 0.9261 | 0.9262 | 0.9261 | 0.9261 | 0.9261 | 0.9261 | 0.9790 | 0.9794 | 0.8523 |

Experiment 1 MLP had the highest accuracy, macro-F1, ROC-AUC, PR-AUC, and MCC. The differences were small: its accuracy was approximately 0.11 percentage points above the baseline and approximately 0.22 percentage points above the attention model.

For this single-label binary classification task, the micro precision, recall, and F1 values equal overall accuracy. Full macro, micro, and weighted precision/recall/F1 values are recorded below.

### Precision, recall, and F1 by averaging method

| Model | P macro | R macro | F1 macro | P micro | R micro | F1 micro | P weighted | R weighted | F1 weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Mean+Max | 0.9272 | 0.9272 | 0.9272 | 0.9272 | 0.9272 | 0.9272 | 0.9272 | 0.9272 | 0.9272 |
| Experiment 1 MLP | 0.9285 | 0.9283 | 0.9283 | 0.9283 | 0.9283 | 0.9283 | 0.9285 | 0.9283 | 0.9283 |
| Experiment 2 Attention | 0.9262 | 0.9261 | 0.9261 | 0.9261 | 0.9261 | 0.9261 | 0.9262 | 0.9261 | 0.9261 |

### Confusion matrices

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Baseline Mean+Max | 17,549 | 1,451 | 1,317 | 17,683 |
| Experiment 1 MLP | 17,861 | 1,139 | 1,586 | 17,414 |
| Experiment 2 Attention | 17,475 | 1,525 | 1,282 | 17,718 |

Experiment 1 produced fewer false positives than the other two models, but more false negatives than the baseline and attention model. Confusion-matrix images are saved at:

- `task2_sentiment/jiayi/outputs/baseline_confusion_matrix.png`
- `task2_sentiment/jiayi/outputs/experiment_1_mlp_confusion_matrix.png`
- `task2_sentiment/jiayi/outputs/experiment_2_attention_confusion_matrix.png`

### Calibration

| Model | Brier score | Expected calibration error |
|---|---:|---:|
| Baseline Mean+Max | 0.05384 | 0.00955 |
| Experiment 1 MLP | 0.05331 | 0.00932 |
| Experiment 2 Attention | 0.05457 | 0.00925 |

Lower is better for both measures. Experiment 1 had the lowest Brier score, while Experiment 2 had the lowest ECE by a small margin. All three ECE values were below 0.01.

### 95% bootstrap confidence intervals

The intervals were computed with 1,000 bootstrap resamples using seed `4253`.

| Model | Accuracy 95% CI | Macro-F1 95% CI | MCC 95% CI |
|---|---|---|---|
| Baseline Mean+Max | [0.9246, 0.9297] | [0.9245, 0.9297] | [0.8491, 0.8594] |
| Experiment 1 MLP | [0.9257, 0.9310] | [0.9257, 0.9310] | [0.8517, 0.8623] |
| Experiment 2 Attention | [0.9236, 0.9288] | [0.9236, 0.9288] | [0.8473, 0.8577] |

The confidence intervals overlap substantially across all three models, consistent with the small differences in their point estimates.

### Paired McNemar tests

Each experimental model was compared against the baseline using paired predictions on the same 38,000 test examples.

| Comparison | Baseline correct / Exp wrong | Baseline wrong / Exp correct | Net experimental wins | p-value | Significant at 0.05 |
|---|---:|---:|---:|---:|---|
| Baseline vs. Experiment 1 MLP | 538 | 581 | 43 | 0.2093 | No |
| Baseline vs. Experiment 2 Attention | 784 | 745 | -39 | 0.3311 | No |

Neither experimental model differed significantly from the baseline at the 0.05 level. Therefore, Experiment 1 is the best model numerically, but the paired test does not provide evidence that its prediction accuracy is significantly different from the baseline on this test set.

## Robustness by review-length slice

Review-length slices were defined from the raw training-review distribution:

- **Short:** <= 97 words
- **Medium:** 98-282 words
- **Long:** > 282 words

| Model | Short n | Short Macro-F1 | Short Error | Medium n | Medium Macro-F1 | Medium Error | Long n | Long Macro-F1 | Long Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Mean+Max | 19,040 | 0.9248 | 0.0738 | 15,214 | 0.9298 | 0.0697 | 3,746 | 0.9102 | 0.0804 |
| Experiment 1 MLP | 19,040 | 0.9266 | 0.0724 | 15,214 | 0.9299 | 0.0694 | 3,746 | 0.9124 | 0.0777 |
| Experiment 2 Attention | 19,040 | 0.9267 | 0.0720 | 15,214 | 0.9247 | 0.0749 | 3,746 | 0.9121 | 0.0793 |

Long reviews were the most difficult slice for all three models by error rate. Experiment 1 had the best long-review macro-F1 (0.9124) and the lowest long-review error rate (0.0777). Experiment 2 performed slightly best on the short slice but dropped more noticeably on medium reviews.

## Manual error review

The 20-case audit set was generated from **Experiment 1 MLP**, the strongest model by overall test metrics. The selected cases follow the required structure:

| Review group | Count |
|---|---:|
| Confident false positives | 5 |
| Confident false negatives | 5 |
| Near-threshold errors | 5 |
| Slice-specific failures | 5 |
| **Total** | **20** |

For the slice-specific group, the automatically identified worst-performing slice for Experiment 1 was the **long-review** slice. The selected review cases are saved at:

`task2_sentiment/jiayi/outputs/manual_error_review_20.csv`

Case-level qualitative error types, observations, and proposed testable fixes belong in the separate `failure_analysis.md` rather than being duplicated here.

## Efficiency and hardware

| Metric | Baseline Mean+Max | Experiment 1 MLP | Experiment 2 Attention |
|---|---:|---:|---:|
| Parameter count | 3,840,257 | 3,850,369 | 3,892,417 |
| Training time (s) | 533.65 | 612.58 | 586.06 |
| Training examples/sec | 4722.18 | 4113.73 | 4299.92 |
| Peak GPU memory (MB) | 89.90 | 90.79 | 110.75 |

All three official runs used:

- CPU: `Intel(R) Xeon(R) CPU @ 2.20GHz`
- GPU: `NVIDIA A100-SXM4-40GB`
- Python: `3.13.15`
- PyTorch: `2.11.0+cu130`
- CUDA: `13.0`

The baseline was the fastest and used the least GPU memory. The attention model used the most parameters and peak GPU memory. Experiment 1 achieved the strongest overall classification metrics but required the longest training time.

## Observations and shortcomings

1. **Experiment 1 MLP was the strongest overall model numerically.** It achieved the highest accuracy (0.9283), macro-F1 (0.9283), ROC-AUC (0.9796), PR-AUC (0.9797), and MCC (0.8568).
2. **The improvement over the baseline was small and not statistically significant.** The McNemar p-value for Baseline vs. Experiment 1 was 0.2093.
3. **Single-head self-attention did not improve overall classification performance.** Experiment 2 had slightly lower accuracy and macro-F1 than the simpler baseline, despite using more parameters and GPU memory.
4. **All three models showed mild late-epoch overfitting.** Training loss continued decreasing while validation loss rose after the best checkpoint.
5. **Long reviews were consistently the hardest length slice.** All models had their highest slice error rate on reviews longer than 282 raw words.
6. **Calibration was similar across the three models.** Experiment 1 had the best Brier score, while Experiment 2 had the lowest ECE by a small margin.
7. **Model differences are modest.** The overlapping bootstrap confidence intervals and non-significant McNemar tests suggest that the architectures produced broadly similar final performance on this dataset.

Possible future improvements include tuning regularization and early stopping, testing alternative vocabulary and sequence-length settings, and improving the handling of long or mixed-sentiment reviews while keeping the no-pretrained-model constraint.

## Evidence and reproducibility

### Baseline Mean+Max

Run ID: `20261004_234803`

Checkpoint:

`task2_sentiment/jiayi/checkpoints/task2_baseline_meanmax_best_20261004_234803.pt`

Raw training log:

`reproducibility/raw_logs/task2_jiayi_baseline_meanmax_train_20261004_234803.log`

Manifest:

`reproducibility/manifests/task2_jiayi_baseline_meanmax_manifest_20261004_234803.json`

Loss curve:

`task2_sentiment/jiayi/outputs/baseline_meanmax_loss_curve_20261004_234803.png`

### Experiment 1 MLP

Run ID: `20261002_211627`

Checkpoint:

`task2_sentiment/jiayi/checkpoints/task2_experiment1_mlp_best_20261002_211627.pt`

Raw training log:

`reproducibility/raw_logs/task2_jiayi_experiment1_mlp_train_20261002_211627.log`

Manifest:

`reproducibility/manifests/task2_jiayi_experiment1_mlp_manifest_20261002_211627.json`

Loss curve:

`task2_sentiment/jiayi/outputs/experiment1_mlp_loss_curve_20261002_211627.png`

### Experiment 2 Attention

Run ID: `20261003_135109`

Checkpoint:

`task2_sentiment/jiayi/checkpoints/task2_experiment2_attention_best_20261003_135109.pt`

Raw training log:

`reproducibility/raw_logs/task2_jiayi_experiment2_attention_train_20261003_135109.log`

Manifest:

`reproducibility/manifests/task2_jiayi_experiment2_attention_manifest_20261003_135109.json`

Loss curve:

`task2_sentiment/jiayi/outputs/experiment2_attention_loss_curve_20261003_135109.png`
