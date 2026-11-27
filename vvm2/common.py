"""Shared pieces: the ordinary data sets, an ordinary net to compare against, and the
training loop. Nothing here is part of the machine.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, outputs, width, depth=2):
        super().__init__()
        dims = [in_features] + [width] * depth
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        self.net = nn.Sequential(*layers, nn.Linear(width, outputs))
        self.dims = dims + [outputs]

    def forward(self, x):
        return self.net(x)

    def madds_total(self):
        return sum(a * b + b for a, b in zip(self.dims, self.dims[1:]))


def params(m):
    return sum(p.numel() for p in m.parameters())


def mlp_matched(in_features, outputs, target, depth=2):
    """The MLP whose parameter count is nearest the target."""
    best = None
    for width in range(2, 4097):
        n = params(MLP(in_features, outputs, width, depth))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (width, n)
        if n > target:
            break
    return MLP(in_features, outputs, best[0], depth)


# ---- tasks: each returns train x, y, test x, y, and "class" or "real" ----

def xor(seed):
    """The four corners, and noisy copies of them to test on (noise 0.15)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([0, 1, 1, 0])
    tx = x.repeat(250, 1) + 0.15 * torch.randn(1000, 2, generator=g)
    return x, y, tx, y.repeat(250), "class"


def sin(seed):
    """y = sin(x) on [-pi, pi]: 200 random training points, a grid of 1,000 to test."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(200, 1, generator=g) * 2 - 1) * math.pi
    tx = torch.linspace(-math.pi, math.pi, 1000).unsqueeze(1)
    return x, torch.sin(x), tx, torch.sin(tx), "real"


def spirals(seed):
    """Two interleaved spirals, 1,000 training points and 1,000 fresh ones to test."""
    g = torch.Generator().manual_seed(seed)

    def draw(n):
        t = torch.rand(n, generator=g) * 3 * math.pi + 0.5
        c = torch.randint(2, (n,), generator=g)
        r = t / (3 * math.pi)
        a = t + math.pi * c
        x = torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=1)
        return x + 0.02 * torch.randn(n, 2, generator=g), c

    x, y = draw(1000)
    tx, ty = draw(1000)
    return x, y, tx, ty, "class"


def mnist(seed, folder="data/mnist"):
    """The standard 60,000 / 10,000 split, pixels scaled to 0..1 (scripts/get_mnist.sh)."""
    def idx(name, offset, shape):
        raw = np.fromfile(os.path.join(folder, name), dtype=np.uint8)[offset:]
        return torch.from_numpy(raw.reshape(shape).copy())

    x = idx("train-images-idx3-ubyte", 16, (-1, 784)).float() / 255
    y = idx("train-labels-idx1-ubyte", 8, (-1,)).long()
    tx = idx("t10k-images-idx3-ubyte", 16, (-1, 784)).float() / 255
    ty = idx("t10k-labels-idx1-ubyte", 8, (-1,)).long()
    return x, y, tx, ty, "class"


def kmnist(seed):
    """Kuzushiji-MNIST: ten handwritten Japanese characters, same format as MNIST
    (scripts/get_data.sh kmnist)."""
    return mnist(seed, "data/kmnist")


def fashion(seed):
    """Fashion-MNIST: ten kinds of clothing, shoes and bags (scripts/get_data.sh fashion)."""
    return mnist(seed, "data/fashion")


TASKS = {"xor": xor, "sin": sin, "spirals": spirals, "mnist": mnist, "kmnist": kmnist,
         "fashion": fashion}


def score(model, x, y, kind):
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(x[i:i + 2000]) for i in range(0, len(x), 2000)])
    if kind == "class":
        return (out.argmax(dim=1) == y).float().mean().item()
    return F.mse_loss(out, y).item()


def shifted(x, most, g):
    """Each 28 by 28 image moved by up to `most` pixels each way, at random."""
    n = len(x)
    img = F.pad(x.view(n, 28, 28), (most, most, most, most))
    oy = torch.randint(2 * most + 1, (n, 1, 1), generator=g)
    ox = torch.randint(2 * most + 1, (n, 1, 1), generator=g)
    rows = oy + torch.arange(28).view(1, 28, 1)
    cols = ox + torch.arange(28).view(1, 1, 28)
    return img[torch.arange(n).view(n, 1, 1), rows, cols].reshape(n, 784)


def train(model, data, steps, batch, lr, seed, wd=0.0, shift=0, augment=None, loss_fn=None,
          each_step=None):
    """wd: weight decay (AdamW). shift: MNIST only, move each training image by up to this
    many pixels at random (the usual augmentation)."""
    x, y, tx, ty, kind = data
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    start = time.time()
    for step in range(steps):
        model.train()
        if each_step is not None:
            each_step(model, step / steps)   # for schedules: the share of training done
        i = torch.randint(len(x), (min(batch, len(x)),), generator=g)
        xb = shifted(x[i], shift, g) if shift else x[i]
        if augment is not None:
            xb = augment(xb, g)
        if loss_fn is not None:
            loss = loss_fn(model, xb, y[i])
        else:
            out = model(xb)
            loss = F.cross_entropy(out, y[i]) if kind == "class" else F.mse_loss(out, y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return score(model, x, y, kind), score(model, tx, ty, kind), time.time() - start


def patches(x):
    """[B, 784] image -> [B, 49, 16]: the 4 by 4 patches, row by row."""
    img = x.view(-1, 7, 4, 7, 4).permute(0, 1, 3, 2, 4)
    return img.reshape(-1, 49, 16)
