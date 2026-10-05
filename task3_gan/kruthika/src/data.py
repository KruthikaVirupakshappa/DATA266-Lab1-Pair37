"""Task 3.1 - the two unpaired domains (A = Monet, B = photo) and augmentation.

All images are loaded once as uint8 tensors (300 + 7,038 images, ~1.4 GB) and kept on the
GPU, so a training batch is just random indexing - no data-loader bottleneck.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def list_images(folder):
    return sorted(Path(folder).glob("*.jpg"))


def load_images(folder):
    """-> (uint8 tensor [N, 3, 256, 256], list of file names), in sorted file-name order."""
    files = list_images(folder)
    arr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous(), [f.name for f in files]


def to_float(x_uint8):
    """uint8 [0, 255] -> float [-1, 1]."""
    return x_uint8.float().div_(127.5).sub_(1.0)


def to_uint8(x):
    """float [-1, 1] -> uint8 [0, 255]."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def sample_batch(images, batch_size, jitter=True):
    """Random batch with the CycleGAN paper's jitter: random horizontal flip and a random
    256x256 crop from the image upscaled to 286x286."""
    idx = torch.randint(0, images.shape[0], (batch_size,), device=images.device)
    x = to_float(images[idx])
    if jitter:
        flip = torch.rand(batch_size, device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        x = F.interpolate(x, size=286, mode="bilinear", align_corners=False)
        i, j = np.random.randint(0, 286 - 256 + 1, size=2)
        x = x[:, :, i:i + 256, j:j + 256]
    return x


# ---------------- DiffAugment (Zhao et al., 2020) ----------------
# Applied to BOTH real and fake images right before the discriminator, so the discriminator
# cannot memorise the 300 Monets; gradients still flow to the generator. Never applied to
# the images we submit.
def _color(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)                       # brightness
    m = x.mean(dim=1, keepdim=True)
    x = (x - m) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2) + m                # saturation
    m = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - m) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + m              # contrast
    return x


def _translation(x, ratio=0.125):
    B, _, H, W = x.shape
    sx, sy = int(H * ratio + 0.5), int(W * ratio + 0.5)
    tx = torch.randint(-sx, sx + 1, (B, 1, 1), device=x.device)
    ty = torch.randint(-sy, sy + 1, (B, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(B, device=x.device), torch.arange(H, device=x.device),
                                torch.arange(W, device=x.device), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, H + 1)
    gy = torch.clamp(gy + ty + 1, 0, W + 1)
    padded = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return padded.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def _cutout(x, ratio=0.5):
    B, _, H, W = x.shape
    ch, cw = int(H * ratio + 0.5), int(W * ratio + 0.5)
    ox = torch.randint(0, H + (1 - ch % 2), (B, 1, 1), device=x.device)
    oy = torch.randint(0, W + (1 - cw % 2), (B, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(B, device=x.device), torch.arange(ch, device=x.device),
                                torch.arange(cw, device=x.device), indexing="ij")
    gx = torch.clamp(gx + ox - ch // 2, 0, H - 1)
    gy = torch.clamp(gy + oy - cw // 2, 0, W - 1)
    mask = torch.ones(B, H, W, dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


_POLICY = {"color": _color, "translation": _translation, "cutout": _cutout}


def diff_augment(x, policy):
    """policy: e.g. 'color,translation,cutout'; empty string = no augmentation."""
    for p in [p for p in policy.split(",") if p]:
        x = _POLICY[p](x)
    return x
