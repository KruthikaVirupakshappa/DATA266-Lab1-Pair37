"""Task 3 evaluation metrics.

The competition score follows the instructor's evaluation script (Part3_Evaluation_Script.ipynb):
- Inception-v3 (torchvision IMAGENET1K_V1, transform_input=False), 2048-d pool features;
  images go through PIL Resize(299) -> CenterCrop(299) -> ToTensor -> ImageNet normalisation.
- Only the first N_EVAL = 300 files (sorted by name) of each folder are used.
- FID between real and generated features; "MiFID" = mean cosine distance between real[i] and
  generated[i], paired by index.
- Done for both directions (photo->Monet vs real Monets, Monet->photo vs real photos), and the
  submission reports the AVERAGE FID and AVERAGE MiFID of the two directions.
`official_features` reproduces that pipeline exactly (used for submission.csv);
`InceptionFeatures.__call__` is a fast GPU version for scoring during training (cosine 0.9998 with it).
Pretrained networks are used here ONLY to measure images, never to generate them.
"""
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg

N_EVAL = 300
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class InceptionFeatures:
    def __init__(self, device):
        import torchvision
        self.model = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)
        self.device = device
        self.mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images_uint8, batch_size=100):
        """Fast path: uint8 [N, 3, 256, 256] -> float32 numpy [N, 2048]."""
        out = []
        for i in range(0, len(images_uint8), batch_size):
            x = images_uint8[i:i + batch_size].to(self.device).float() / 255.0
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            out.append(self.model((x - self.mean) / self.std).float().cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def official_features(self, paths, batch_size=32):
        """Exactly the evaluation script's pipeline, on image files."""
        import torchvision.transforms as T
        tf = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor(), T.Normalize(_MEAN, _STD)])
        feats = []
        for i in range(0, len(paths), batch_size):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]).to(self.device)
            feats.append(self.model(x).detach().cpu().numpy())
        return np.concatenate(feats, axis=0)


def jpeg_roundtrip(images_uint8, quality):
    """Encode/decode as JPEG in memory, so in-training scores see the same compression as the files."""
    out = []
    for img in images_uint8.cpu():
        buf = io.BytesIO()
        Image.fromarray(img.permute(1, 2, 0).numpy()).save(buf, format="JPEG", quality=quality)
        out.append(torch.from_numpy(np.asarray(Image.open(buf).convert("RGB"))).permute(2, 0, 1))
    return torch.stack(out)


# ---------------- competition metrics (as in the evaluation script) ----------------
def gaussian_stats(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


def fid(mu1, s1, mu2, s2, eps=1e-6):
    """Same maths as the script's frechet_distance; works with old and new scipy versions
    (scipy >= 1.18 removed the `disp` argument the script passes to sqrtm)."""
    covmean = linalg.sqrtm(s1.dot(s2))
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(s1 + s2 - 2 * covmean))


def fid_mifid(real_feats, gen_feats):
    """One direction, as calculate_fid_mifid in the script: match counts by taking the first n,
    FID of the two sets, MiFID = mean cosine distance of real[i] vs gen[i]."""
    n = min(len(real_feats), len(gen_feats))
    r, g = real_feats[:n].astype(np.float64), gen_feats[:n].astype(np.float64)
    f = fid(*gaussian_stats(r), *gaussian_stats(g))
    cos = (r * g).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(g, axis=1))
    return f, float((1 - cos).mean())


def competition_score(real_monet_f, real_photo_f, gen_b2a_f, gen_a2b_f):
    """Both directions + submission values (average FID, average MiFID) + estimated leaderboard score."""
    fid_b2a, mifid_b2a = fid_mifid(real_monet_f[:N_EVAL], gen_b2a_f[:N_EVAL])
    fid_a2b, mifid_a2b = fid_mifid(real_photo_f[:N_EVAL], gen_a2b_f[:N_EVAL])
    sub_fid, sub_mifid = (fid_a2b + fid_b2a) / 2, (mifid_a2b + mifid_b2a) / 2
    return {"fid_B2A": fid_b2a, "mifid_B2A": mifid_b2a, "fid_A2B": fid_a2b, "mifid_A2B": mifid_a2b,
            "FID": sub_fid, "MiFID": sub_mifid, "score": (sub_fid + sub_mifid) / 2}


def _cos_dist(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - a @ b.T


# ---------------- additional metrics (brief) ----------------
def kid(feats_gen, feats_real, n_subsets=100, subset_size=1000, seed=0):
    """Kernel Inception Distance: unbiased MMD^2 with a cubic polynomial kernel, mean and std over subsets."""
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(feats_gen), len(feats_real))
    d = feats_gen.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = feats_gen[rng.choice(len(feats_gen), m, replace=False)].astype(np.float64)
        y = feats_real[rng.choice(len(feats_real), m, replace=False)].astype(np.float64)
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def prdc(feats_real, feats_gen, k=5, max_n=3000, seed=0):
    """Precision, recall, density, coverage (Naeem et al., 2020) with k-NN balls in feature space."""
    rng = np.random.default_rng(seed)
    if len(feats_real) > max_n:
        feats_real = feats_real[rng.choice(len(feats_real), max_n, replace=False)]
    if len(feats_gen) > max_n:
        feats_gen = feats_gen[rng.choice(len(feats_gen), max_n, replace=False)]
    r, g = torch.from_numpy(feats_real).float(), torch.from_numpy(feats_gen).float()
    d_rr, d_gg, d_rg = torch.cdist(r, r), torch.cdist(g, g), torch.cdist(r, g)
    rad_r = d_rr.kthvalue(k + 1, dim=1).values  # +1 skips the point itself
    rad_g = d_gg.kthvalue(k + 1, dim=1).values
    inside = d_rg < rad_r[:, None]                # [real, gen]: gen inside real's ball
    return {
        "precision": inside.any(0).float().mean().item(),
        "recall": (d_rg < rad_g[None, :]).any(1).float().mean().item(),
        "density": (inside.sum(0).float() / k).mean().item(),
        "coverage": (d_rg.min(1).values < rad_r).float().mean().item(),
    }


def content_cosine(feats_in, feats_out):
    """Mean cosine similarity between Inception features of each input and its translation."""
    return float(1.0 - np.diag(_cos_dist(feats_in, feats_out)).mean())


class LPIPS:
    def __init__(self, device):
        import lpips
        self.model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, a_uint8, b_uint8, batch_size=50):
        vals = []
        for i in range(0, len(a_uint8), batch_size):
            a = a_uint8[i:i + batch_size].to(self.device).float() / 127.5 - 1
            b = b_uint8[i:i + batch_size].to(self.device).float() / 127.5 - 1
            vals.append(self.model(a, b).flatten().cpu())
        return float(torch.cat(vals).mean())


def cohen_kappa(r1, r2):
    """Agreement between two raters beyond chance (labels can be any hashable values)."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    labels = np.unique(np.concatenate([r1, r2]))
    po = (r1 == r2).mean()
    pe = sum((r1 == c).mean() * (r2 == c).mean() for c in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
