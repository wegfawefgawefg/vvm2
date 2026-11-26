"""Baselines: an MLP, a GRU and a small transformer, each sizable by one width.

All take x [B, L] token ids and return logits [B, classes]. Each has madds(length): the
multiply-adds for one example, by part, counted the same way as the vvm (one per
weight use in a matrix product, the width of a row per embedding read, the width of a
vector per norm; biases and activations free).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """A linear layer on the concatenated one-hot tokens (done as a sum of embedding
    rows), then depth-1 hidden layers, then the readout. If the task's tokens are shared
    across positions, each position gets its own rows; otherwise the token ids already
    say the position."""

    def __init__(self, vocab, length, classes, width, depth=2, shared_vocab=False):
        super().__init__()
        self.vocab, self.width, self.classes = vocab, width, classes
        self.first = nn.Embedding(length * vocab if shared_vocab else vocab, width)
        self.bias = nn.Parameter(torch.zeros(width))
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.out = nn.Linear(width, classes)
        self.register_buffer("offset", torch.arange(length) * (vocab if shared_vocab else 0))

    def forward(self, x):
        h = self.first(x + self.offset).sum(dim=1) + self.bias
        for layer in self.hidden:
            h = layer(F.gelu(h))
        return self.out(F.gelu(h))

    def madds(self, length):
        w = self.width
        return {"embed": length * w, "hidden": len(self.hidden) * w * w,
                "readout": w * self.classes}


class GRU(nn.Module):
    """Reads the tokens one a step, then `ponder` further steps with a zero input, and
    reads out from the last hidden state. With ponder it is a dense tied-weight vvm."""

    def __init__(self, vocab, length, classes, width, ponder=0):
        super().__init__()
        self.width, self.classes, self.ponder = width, classes, ponder
        self.embed = nn.Embedding(vocab, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.out = nn.Linear(width, classes)

    def forward(self, x):
        e = self.embed(x)
        if self.ponder:
            e = torch.cat([e, e.new_zeros(e.shape[0], self.ponder, e.shape[2])], dim=1)
        _, h = self.gru(e)
        return self.out(h[0])

    def madds(self, length):
        w = self.width
        return {"embed": length * w, "gru": (length + self.ponder) * 6 * w * w,
                "readout": w * self.classes}


class Block(nn.Module):
    def __init__(self, width, heads, ff):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.f1, self.f2 = nn.Linear(width, ff), nn.Linear(ff, width)

    def forward(self, x):
        b, n, w = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, w)
        x = x + self.proj(a)
        return x + self.f2(F.gelu(self.f1(self.n2(x))))


class Transformer(nn.Module):
    """Pre-norm encoder over the L tokens, mean-pooled, then the readout."""

    def __init__(self, vocab, length, classes, width, layers=2, heads=4, ff_mult=2):
        super().__init__()
        self.width, self.classes, self.ff = width, classes, ff_mult * width
        self.embed = nn.Embedding(vocab, width)
        self.pos = nn.Parameter(torch.randn(length, width) * 0.02)
        self.blocks = nn.ModuleList(Block(width, heads, self.ff) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, classes)

    def forward(self, x):
        h = self.embed(x) + self.pos
        for block in self.blocks:
            h = block(h)
        return self.out(self.norm(h.mean(dim=1)))

    def madds(self, length):
        w, n = self.width, length
        per_block = n * 4 * w * w + 2 * n * n * w + n * 2 * w * self.ff + 2 * n * w
        return {"embed": n * w, "blocks": len(self.blocks) * per_block,
                "readout": w * self.classes + w}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_madds(model, length):
    return sum(model.madds(length).values())


def fit_width(make, measure, target, step=4):
    """The width (a multiple of step) whose measure(make(width)) is nearest the target.
    measure must grow with width."""
    hi = 1
    while measure(make(hi * step)) < target:
        hi *= 2
    lo = max(1, hi // 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if measure(make(mid * step)) < target:
            lo = mid + 1
        else:
            hi = mid
    near = [w for w in (lo - 1, lo) if w >= 1]
    return step * min(near, key=lambda w: abs(measure(make(w * step)) - target))
