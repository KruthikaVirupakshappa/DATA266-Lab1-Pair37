# Task 2 model comparison (all six models, identical metric code)

Official Yelp Polarity test split, 38,000 reviews; threshold 0.5. K = Kruthika, J = Jiayi. All embeddings are learned from scratch.

| | K-baseline (mean embedding) | K-exp1 (BiLSTM) | K-exp2 (TextCNN) | J-baseline (mean+max embedding) | J-exp1 (MLP) | J-exp2 (self-attention) |
|---|---|---|---|---|---|---|
| Member | Kruthika | Kruthika | Kruthika | Jiayi | Jiayi | Jiayi |
| Architecture / key hyperparameters | mean of word embeddings (d=128) → linear; dropout 0.3; Adam lr 0.002 | embeddings (d=128) → 1-layer BiLSTM (128/direction) → max+mean pooling → linear; dropout 0.3; lr 0.001 | embeddings (d=128) → Conv1d [3, 4, 5] × 128 filters → max-over-time → linear; dropout 0.5; lr 0.001 | [mean ; max] of word embeddings (d=128) → linear; lr 0.001 | mean of word embeddings (d=128) → MLP 64→32→1; dropout 0.3; lr 0.001 | word + position embeddings (d=128) → single-head self-attention (d=64) → mean pooling → linear; lr 0.001 |
| **Accuracy** [95% CI] | 0.9326 [0.9301, 0.9352] | 0.9531 [0.9509, 0.9550] | 0.9435 [0.9412, 0.9456] | 0.9272 [0.9246, 0.9297] | 0.9283 [0.9257, 0.9309] | 0.9261 [0.9234, 0.9285] |
| Precision macro / micro / weighted | 0.9326 / 0.9326 / 0.9326 | 0.9531 / 0.9531 / 0.9531 | 0.9436 / 0.9435 / 0.9436 | 0.9272 / 0.9272 / 0.9272 | 0.9285 / 0.9283 / 0.9285 | 0.9262 / 0.9261 / 0.9262 |
| Recall macro / micro / weighted | 0.9326 / 0.9326 / 0.9326 | 0.9531 / 0.9531 / 0.9531 | 0.9435 / 0.9435 / 0.9435 | 0.9272 / 0.9272 / 0.9272 | 0.9283 / 0.9283 / 0.9283 | 0.9261 / 0.9261 / 0.9261 |
| F1 micro / weighted | 0.9326 / 0.9326 | 0.9531 / 0.9531 | 0.9435 / 0.9435 | 0.9272 / 0.9272 | 0.9283 / 0.9283 | 0.9261 / 0.9261 |
| **Macro-F1** [95% CI] | 0.9326 [0.9301, 0.9352] | 0.9531 [0.9509, 0.9550] | 0.9435 [0.9412, 0.9456] | 0.9272 [0.9246, 0.9297] | 0.9283 [0.9257, 0.9309] | 0.9261 [0.9234, 0.9285] |
| Confusion matrix TN / FP / FN / TP | 17,799 / 1,201 / 1,360 / 17,640 | 18,059 / 941 / 841 / 18,159 | 17,852 / 1,148 / 998 / 18,002 | 17,549 / 1,451 / 1,317 / 17,683 | 17,861 / 1,139 / 1,586 / 17,414 | 17,475 / 1,525 / 1,282 / 17,718 |
| ROC-AUC | 0.9790 | 0.9902 | 0.9867 | 0.9789 | 0.9796 | 0.9790 |
| PR-AUC (average precision) | 0.9788 | 0.9902 | 0.9871 | 0.9789 | 0.9797 | 0.9794 |
| **MCC** [95% CI] | 0.8652 [0.8602, 0.8703] | 0.9062 [0.9019, 0.9101] | 0.8871 [0.8825, 0.8913] | 0.8543 [0.8492, 0.8595] | 0.8568 [0.8517, 0.8619] | 0.8523 [0.8468, 0.8570] |
| Brier score | 0.0517 | 0.0359 | 0.0424 | 0.0538 | 0.0533 | 0.0546 |
| ECE (15 bins) | 0.0118 | 0.0135 | 0.0049 | 0.0074 | 0.0083 | 0.0046 |
| Slice long (>200) (n=7,466): macro-F1 / error rate | 0.9305 / 6.50% | 0.9535 / 4.38% | 0.9394 / 5.71% | 0.9194 / 7.58% | 0.9214 / 7.34% | 0.9174 / 7.81% |
| Slice medium (51-200) (n=21,172): macro-F1 / error rate | 0.9323 / 6.77% | 0.9547 / 4.53% | 0.9454 / 5.46% | 0.9299 / 7.01% | 0.9300 / 7.00% | 0.9274 / 7.25% |
| Slice short (<=50 words) (n=9,362): macro-F1 / error rate | 0.9285 / 6.87% | 0.9448 / 5.30% | 0.9370 / 6.04% | 0.9201 / 7.66% | 0.9230 / 7.42% | 0.9234 / 7.35% |
| Slice has negation (n=28,177): macro-F1 / error rate | 0.9249 / 7.25% | 0.9504 / 4.81% | 0.9404 / 5.77% | 0.9192 / 7.82% | 0.9197 / 7.73% | 0.9172 / 8.03% |
| Slice no negation (n=9,823): macro-F1 / error rate | 0.9274 / 5.28% | 0.9407 / 4.35% | 0.9272 / 5.28% | 0.9211 / 5.74% | 0.9249 / 5.57% | 0.9237 / 5.54% |
| Slice has contrast word (n=22,075): macro-F1 / error rate | 0.9241 / 7.51% | 0.9484 / 5.11% | 0.9382 / 6.13% | 0.9178 / 8.15% | 0.9186 / 8.04% | 0.9151 / 8.43% |
| Slice no contrast word (n=15,925): macro-F1 / error rate | 0.9422 / 5.68% | 0.9582 / 4.10% | 0.9492 / 4.98% | 0.9380 / 6.08% | 0.9393 / 5.97% | 0.9394 / 5.95% |
| Parameters | 5,140,481 | 5,405,057 | 5,337,729 | 3,840,257 | 3,850,369 | 3,892,417 |
| Training time (s) | 25 | 460 | 53 | 534 | 613 | 586 |
| Training examples/s | 98,985 | 5,481 | 47,450 | 4,722 | 4,114 | 4,300 |
| Inference examples/s | 1,187,092 | 62,306 | 203,973 | not reported | not reported | not reported |
| Peak GPU memory (MB) | 965.2 | 1,471.2 | 1,223.9 | 89.9 | 90.8 | 110.7 |

### Paired McNemar tests

| Model A | Model B | A right, B wrong | A wrong, B right | χ² | p-value |
|---|---|---|---|---|---|
| K-baseline (mean embedding) | K-exp1 (BiLSTM) | 588 | 1,367 | 309.61 | 2.66e-69 |
| K-baseline (mean embedding) | K-exp2 (TextCNN) | 677 | 1,092 | 96.89 | 7.33e-23 |
| J-baseline (mean+max embedding) | J-exp1 (MLP) | 538 | 581 | 1.58 | 0.209 |
| J-baseline (mean+max embedding) | J-exp2 (self-attention) | 784 | 745 | 0.94 | 0.331 |
| J-baseline (mean+max embedding) | K-baseline (mean embedding) | 627 | 834 | 29.05 | 7.07e-08 |
| J-exp1 (MLP) | K-exp1 (BiLSTM) | 660 | 1,603 | 392.12 | 2.86e-87 |

## How to read this table

- **Identical code and data:** every quality metric above is recomputed by this script from the two members'
  `test_predictions.csv` files (official test split, 38,000 reviews, same order and labels) with the same settings.
  The sanity check confirms that the recomputed accuracies equal each member's reported accuracies exactly.
- **Different settings in the members' own reports:** Jiayi's notebook computed ECE with 10 bins over P(positive)
  and PR-AUC as the trapezoidal area under the precision-recall curve, and used different length slices. Kruthika's
  computed ECE with 15 bins over the confidence of the predicted class and PR-AUC as average precision. This table
  uses one definition for all six models, so ECE, PR-AUC and slice values here can differ slightly from the
  individual reports.
- **Slices** use the same definitions for all models: review length in raw words (≤50 / 51–200 / >200),
  whether the review contains a negation, and whether it contains a contrast word (but / however / although /
  though / yet).
- **Efficiency rows** come from each member's own training run (Google Colab GPU; exact GPU in each member's
  metrics report). They are not measured identically: Kruthika's runs use batch size 256 and keep the whole
  encoded dataset on the GPU (about 0.6 GB, which dominates her peak memory); Jiayi's use batch size 64 with a
  data loader. Compare them as orders of magnitude.
- **McNemar** uses the chi-squared test with continuity correction on the same 38,000 reviews; p < 0.05 means the
  two models' error patterns differ significantly.
