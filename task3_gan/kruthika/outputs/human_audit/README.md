# Blinded human audit (30 fixed samples)

Each `sample_XX.jpg` shows the **input (left)** and the **model output (right)**. The translation direction is hidden; do not open `audit_key_DO_NOT_SHOW_RATERS.csv`.

Each rater fills in their own `ratings_raterN.csv` independently, scoring every sample 1-5:
- **style**: does the output look like the target domain (a Monet painting or a real photo)? 1 = not at all, 5 = convincing
- **content**: is the input's scene/structure preserved? 1 = lost, 5 = fully preserved
- **artifacts**: visual defects (blotches, checkerboard, colour bleeding)? 1 = severe, 5 = none

Then run `python task3_gan/kruthika/evaluate_local.py --audit` to compute agreement.
