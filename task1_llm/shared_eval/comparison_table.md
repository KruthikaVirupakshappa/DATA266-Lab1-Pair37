# Task 1 model comparison (shared test set)

Shared test set: 5000 official-validation stories (4,183,390 characters) that are in neither member's training or validation data (see `test_set.json`). Each model uses its own preprocessing, vocabulary and context length; generation uses identical prompts, seeds and settings (4 samples at T=0.8 + 1 greedy per prompt, 300 new characters).

| Metric | Kruthika | Jiayi |
|---|---|---|
| Architecture | 6 layers, 6 heads, d=384, context 256 | 2 layers, 4 heads, d=128, context 128 |
| Vocabulary / preprocessing | 89 chars, cleaned text | 152 chars, raw text |
| Test cross-entropy (nats) | 0.5373 | 0.9243 |
| Test perplexity | 1.711 | 2.520 |
| Test bits-per-character | 0.7751 | 1.3334 |
| **Test bits per original character** | 0.7756 | 1.3350 |
| Test top-1 accuracy | 0.8279 | 0.7081 |
| Distinct-1 / 2 / 3 (words, T=0.8) | 0.338 / 0.745 / 0.908 | 0.390 / 0.812 / 0.947 |
| Repeated 4-gram rate (T=0.8 / greedy) | 0.0073 / 0.0348 | 0.0009 / 0.4203 |
| Generation speed (chars/s, CPU) | 138.8 | 1,581 |
| *From each member's own training run:* | | |
| parameters | 10,814,297 | 452,248 |
| epochs | 10 | 10 |
| train_cross_entropy | 0.5183 | 0.9253 |
| val_cross_entropy | 0.5356 | 0.9296 |
| val_perplexity | 1.7084 | 2.5335 |
| val_bits_per_char | 0.7726 | 1.3411 |
| val_top1_acc | 0.8281 | 0.7061 |
| generalization_gap | 0.0173 | 0.0043 |
| grad_norm_mean | 0.1651 | 0.8471 |
| grad_norm_max | 1.1290 | 2.9612 |
| nan_inf_steps | 0 | 0 |
| train_tokens_per_s | 946,940 | 572,170 |
| total_train_time_min | 15.7910 | 26.9028 |
| peak_gpu_mem_mb | 5,282 | 123.5 |
| device | NVIDIA A100-SXM4-40GB | NVIDIA A100-SXM4-40GB |

## How to read this table

- **Sanity check:** before scoring the test set, each model was re-scored on its own original validation split and
  reproduced its owner's reported validation loss (difference < 0.0001), so both models and vocabularies were loaded
  correctly (`comparison_report.csv`, first rows).
- **Bits per original character** divides each model's total bits by the length of the raw stories, so it is not
  affected by the two preprocessing pipelines (cleaning changes about 0.06% of characters; 2 characters were unknown
  to the raw-text model and skipped). Plain bits-per-character is almost identical here.
- **Distinct-n rewards incoherence as well as variety:** sentences that jump between unrelated ideas produce many
  unique word combinations. Read it together with the samples (`generation_samples.txt`); a higher value is not
  automatically better.
- **Generation speed** was measured for both models on the same CPU with the same sampling loop. It mostly reflects
  model size (10.8M vs 0.45M parameters) and context length (256 vs 128).
- **Own-training-run rows** come from each member's `metrics_report.csv`: validation numbers there are on each
  member's *own* validation split (different stories), and throughput/memory depend on each member's batch size and
  measurement code. Only the shared-test-set rows are measured on identical data with identical code.
