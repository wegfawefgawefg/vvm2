"""The ordinary tasks: xor, a sine curve, two spirals, MNIST. Real-valued inputs.

  python -m vvm2.classic xor
  python -m vvm2.classic sin
  python -m vvm2.classic spirals
  python -m vvm2.classic mnist --steps 3000

Each line trains the vvm and an MLP of the same parameter count on the same data with
the same optimiser and prints one row a model: parameters, multiply-adds per example, the
train and the held-out score, and for the vvm how its bank was used. The vvm here is a
state and a bank of ops and nothing else: each tick the state forms an address, the nearest
few ops are found and applied to the state. The input is written into the state through a
linear map and the answer read out through another; the "no bank" row is those two alone.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bank import use_stats
from .vvm import VVM


class RealVVM(VVM):
    def __init__(self, in_features, outputs, **kw):
        super().__init__(vocab=1, classes=outputs, **kw)
        self.in_features = in_features
        self.proj = nn.Linear(in_features, self.d)

    def inputs(self, x):
        return [self.proj(x)]

    def madds_total(self):
        parts = self.madds(0)
        return sum(parts.values()) + self.in_features * self.d


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


def train(model, data, steps, batch, lr, seed, wd=0.0, shift=0):
    """wd: weight decay (AdamW). shift: MNIST only, move each training image by up to this
    many pixels at random (the usual augmentation)."""
    x, y, tx, ty, kind = data
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    start = time.time()
    for _ in range(steps):
        model.train()
        i = torch.randint(len(x), (min(batch, len(x)),), generator=g)
        xb = shifted(x[i], shift, g) if shift else x[i]
        out = model(xb)
        loss = F.cross_entropy(out, y[i]) if kind == "class" else F.mse_loss(out, y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return score(model, x, y, kind), score(model, tx, ty, kind), time.time() - start


def bank_use(model, tx):
    """Share of entries picked at all on the test inputs, effective number of entries, the
    top entry's share of the read weight, and the mean overlap of two inputs' picks."""
    b = model.bank
    b.reset_counts()
    b.counting = True
    model.eval()
    with torch.no_grad():
        model(tx[:2000])
        s1, s2 = b.scores(model.proj(tx[:400]).unsqueeze(1).flatten(1))
        idx, _ = b.pick(s1, s2)
    b.counting = False
    st = use_stats(b.mass, b.hits)
    first = idx[:, 0]  # head 0, tick 0: [n, k]
    half = len(first) // 2
    same = (first[:half].unsqueeze(2) == first[half:2 * half].unsqueeze(1)).any(dim=2)
    return (sum(s["used"] for s in st) / len(st), sum(s["eff"] for s in st) / len(st),
            max(s["top"] for s in st), same.float().mean().item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    a = p.add_argument
    a("task", choices=list(TASKS))
    a("--seeds", type=int, default=3)
    a("--steps", type=int, default=2000)
    a("--batch", type=int, default=128)
    a("--lr", type=float, default=3e-3)
    a("--d", type=int, default=64)
    a("--ticks", type=int, default=2)
    a("--entries", type=int, default=1024)
    a("--heads", type=int, default=2)
    a("--k", type=int, default=8)
    a("--snoise", type=float, default=0.0, help="noise on the state, in its rms")
    a("--qnoise", type=float, default=0.0, help="noise on the query (the address)")
    a("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    kind = TASKS[args.task](0)[4]
    what = "accuracy" if kind == "class" else "mean squared error"
    print("%s: train / test %s, %d seeds, %d steps" % (args.task, what, args.seeds, args.steps))
    print("| model | params | madds | train | test | bank used | eff. entries | top | "
          "pick overlap | s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    # vvm: the state and the bank of ops, nothing else. "no bank": only the input write and
    # the readout, which is a linear model: what the ops have to beat.
    for name in ["vvm", "no bank (input write + readout only)", "mlp"]:
        rows = []
        for seed in range(args.seeds):
            data = TASKS[args.task](seed)
            ins = data[0].shape[1]
            outs = int(data[1].max()) + 1 if kind == "class" else 1
            torch.manual_seed(seed)

            def make(bank):
                return RealVVM(ins, outs, d=args.d, ticks=args.ticks, entries=args.entries,
                               heads=args.heads, k=args.k, snoise=args.snoise,
                               qnoise=args.qnoise, sparse=False, use_bank=bank)

            if name == "mlp":
                model = mlp_matched(ins, outs, params(make(True)))
            else:
                model = make(name == "vvm")
            tr, te, secs = train(model, data, args.steps, args.batch, args.lr, seed)
            use = bank_use(model, data[2]) if getattr(model, "bank", None) is not None else None
            rows.append((params(model), model.madds_total(), tr, te, use, secs))
        mean = lambda j: sum(r[j] for r in rows) / len(rows)
        fmt = "%.1e" if kind == "real" else "%.3f"
        use = ["-"] * 4
        if rows[0][4] is not None:
            u = [sum(r[4][j] for r in rows) / len(rows) for j in range(4)]
            use = ["%.2f" % u[0], "%.0f" % u[1], "%.3f" % u[2], "%.2f" % u[3]]
        print("| %s | %d | %d | %s | %s | %s | %.0f |" % (
            name, rows[0][0], rows[0][1], fmt % mean(2), fmt % mean(3), " | ".join(use), mean(5)))


if __name__ == "__main__":
    main()
