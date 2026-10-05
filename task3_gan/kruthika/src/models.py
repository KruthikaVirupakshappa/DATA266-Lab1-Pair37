"""Task 3.1 - CycleGAN networks, written from scratch (Zhu et al., 2017).

Generator: ResNet encoder-decoder (c7s1-64, d128, d256, n x R256, u128, u64, c7s1-3).
Discriminator: 70x70 PatchGAN - classifies overlapping 70x70 patches as real/fake,
which focuses it on local texture (brush strokes) rather than global layout.
Images are in [-1, 1].
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """x + F(x): lets the generator keep the photo's structure and only learn the change."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, ngf=64, n_blocks=9, **_):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(2):  # downsample 256 -> 64
            layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        layers += [ResidualBlock(ch) for _ in range(n_blocks)]
        for _ in range(2):  # upsample 64 -> 256
            layers += [nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """PatchGAN: outputs a 30x30 map of real/fake scores for a 256x256 image."""

    def __init__(self, ndf=64, n_layers=3, **_):
        super().__init__()
        layers = [nn.Conv2d(3, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ch = ndf
        for i in range(1, n_layers + 1):
            nxt = ndf * min(2 ** i, 8)
            stride = 2 if i < n_layers else 1
            layers += [nn.Conv2d(ch, nxt, 4, stride=stride, padding=1), nn.InstanceNorm2d(nxt), nn.LeakyReLU(0.2, True)]
            ch = nxt
        layers += [nn.Conv2d(ch, 1, 4, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def init_weights(m):
    """N(0, 0.02) init as in the CycleGAN paper."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ImagePool:
    """History of generated images (Shrivastava et al.): the discriminator sees a mix of current
    and older fakes, which reduces oscillation between G and D."""

    def __init__(self, size=50):
        self.size, self.images = size, []

    def query(self, images):
        if self.size == 0:
            return images
        out = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif torch.rand(1).item() < 0.5:
                i = torch.randint(0, self.size, (1,)).item()
                out.append(self.images[i].clone())
                self.images[i] = img
            else:
                out.append(img)
        return torch.cat(out)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
