# Task 3 model comparison (identical metric code)

Each member's submitted predictions: the first 300 photos → Monet and all 300 Monets → photo. KID: 100 subsets of 100; PRDC: k = 5.

| Metric | Kruthika | Jiayi |
|---|---|---|
| **Kaggle score** (own submission) | -48.3416 | -54.2804 |
| Submission FID (average of both directions) | 96.27 | 108.14 |
| Submission MiFID (average) | 0.4115 | 0.4168 |
| *Photo → Monet (B2A)* | |  |
| FID | 97.38 | 108.66 |
| MiFID | 0.4088 | 0.4099 |
| KID (mean ± std) | 0.0062 ± 0.0013 | 0.0125 ± 0.0018 |
| Precision / recall (k=5) | 0.550 / 0.727 | 0.427 / 0.737 |
| Density / coverage (k=5) | 0.517 / 0.730 | 0.338 / 0.667 |
| Content cosine (input vs output) | 0.812 | 0.753 |
| LPIPS (input vs output) | 0.346 | 0.518 |
| Saturation / contrast (real Monets 0.305 / 0.180) | 0.189 / 0.136 | 0.253 / 0.171 |
| *Monet → photo (A2B)* | |  |
| FID | 95.07 | 107.53 |
| MiFID | 0.4140 | 0.4237 |
| KID (mean ± std) | 0.0141 ± 0.0022 | 0.0218 ± 0.0030 |
| Precision / recall (k=5) | 0.783 / 0.427 | 0.697 / 0.403 |
| Density / coverage (k=5) | 1.349 / 0.927 | 0.959 / 0.837 |
| Content cosine (input vs output) | 0.807 | 0.793 |
| LPIPS (input vs output) | 0.344 | 0.465 |

### Human audit (computed with identical code; **different raters per model**)

| Criterion | Kruthika | Jiayi |
|---|---|---|
| Style: mean photo→Monet / Monet→photo | 4.70 / 4.30 | 3.87 / 1.67 |
| Content: mean photo→Monet / Monet→photo | 4.50 / 4.30 | 3.83 / 3.50 |
| Artifacts: mean photo→Monet / Monet→photo | 4.50 / 4.20 | 3.27 / 3.00 |
| Style: Cohen's κ / quadratic-weighted κ | 0.877 / 0.926 | 0.413 / 0.882 |
| Content: Cohen's κ / quadratic-weighted κ | 0.591 / 0.636 | 0.157 / 0.381 |
| Artifacts: Cohen's κ / quadratic-weighted κ | 0.458 / 0.457 | 0.176 / 0.357 |

## How to read this table

- **Identical code and data for both models:** every number in the main table is computed by this script from the
  saved prediction folders, on the same 300 + 300 images per member (the images the evaluation script scores).
- **Sanity check:** the recomputed submission FID / MiFID matches each member's `submission.csv` (small differences of
  about 0.05 FID come from different PyTorch/Pillow versions, which decode and resize images slightly differently).
- **Kaggle score** is each member's own submission: −(FID + MiFID) / 2 of their `submission.csv`. The team's
  leaderboard entry shows the best of the two.
- **Cycle-reconstruction L1** needs the generators, so it is quoted from each member's own report, on a common scale:
  mean absolute pixel difference on a 0–1 scale. A value computed on images in [−1, 1] is divided by 2.
- **Human audit:** each member's 30 samples were rated by a different pair of raters (Kruthika's: two friends;
  Jiayi's: two family members), so the human scores describe each model separately and are **not comparable
  across models**; agreement (κ) is comparable as a measure of how consistent each pair of raters was.
- **Colour:** real Monets have mean saturation 0.305 and contrast 0.180; values closer to these mean a more Monet-like
  palette.
