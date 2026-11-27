"""vvm on a vector RAM: memory is a list of vectors, and a vop is an instruction on them.

  python -m old.ram [TASK]        default mnist; also kmnist, fashion

Memory is slots of 16 numbers each:
  input     49 slots: the image cut into 4 by 4 patches, one patch a slot. Read-only.
  scratch   16 slots, start empty. Read and written.
  output    1 slot; its first 10 numbers are the answer.

A vop is one instruction:   slot[c] += tanh(M . [slot[a], slot[b]] + bias)
  a, b, c   which two slots it reads and which one it writes: fixed when the vop is made,
            as register numbers are in a machine instruction (c is never an input slot)
  M         16 by 32 numbers: what it does with the two vectors it read
  key       16 numbers: how it is found

One tick: the address is made from the scratch slots; the 32 vops whose keys match best are
fetched and run. The same bank every tick. On the first tick scratch is empty, so every
image fetches the same 32 vops, which read patches of the image into scratch.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classic import TASKS, params, score, train


def patches(x):
    """[B, 784] image -> [B, 49, 16]: the 4 by 4 patches, row by row."""
    img = x.view(-1, 7, 4, 7, 4).permute(0, 1, 3, 2, 4)
    return img.reshape(-1, 49, 16)


class RamVVM(nn.Module):
    def __init__(self, n_in=49, scratch=16, d=16, ops=2048, k=32, ticks=4, d_key=16):
        super().__init__()
        self.n_in, self.scratch, self.d, self.ops, self.k, self.ticks = n_in, scratch, d, ops, k, ticks
        slots = n_in + scratch + 1
        self.d_key = d_key
        self.query = nn.Linear(scratch * d, d_key)
        self.keys = nn.Parameter(torch.randn(ops, d_key) / math.sqrt(d_key))
        self.m = nn.Parameter(torch.randn(ops, d, 2 * d) / math.sqrt(2 * d))
        self.bias = nn.Parameter(torch.zeros(ops, d))
        g = torch.Generator().manual_seed(0)
        self.register_buffer("a", torch.randint(slots, (ops,), generator=g))
        self.register_buffer("b", torch.randint(slots, (ops,), generator=g))
        self.register_buffer("c", n_in + torch.randint(scratch + 1, (ops,), generator=g))
        self.scale = nn.Parameter(torch.tensor(3.0))
        self.register_buffer("picks", torch.zeros(ops))

    def forward(self, x):
        n, d = len(x), self.d
        mem = torch.cat([patches(x), x.new_zeros(n, self.scratch + 1, d)], dim=1)
        for _ in range(self.ticks):
            scr = mem[:, self.n_in:self.n_in + self.scratch].flatten(1)
            q = F.layer_norm(self.query(F.layer_norm(scr, scr.shape[1:])), (self.d_key,))
            sc, idx = (q @ self.keys.T).topk(self.k, dim=-1)                 # [B, k]
            w = torch.softmax(sc, dim=-1) * self.k

            def take(which):
                return mem.gather(1, which[idx].unsqueeze(-1).expand(-1, -1, d))

            seen = torch.cat([take(self.a), take(self.b)], dim=-1)           # [B, k, 2d]
            out = torch.tanh(torch.einsum("bkij,bkj->bki", self.m[idx], seen) + self.bias[idx])
            out = out * w.unsqueeze(-1)
            mem = mem.scatter_add(1, self.c[idx].unsqueeze(-1).expand(-1, -1, d), out)
            if not self.training:
                self.picks += torch.bincount(idx.flatten(), minlength=self.ops).float()
        return mem[:, -1, :10] * self.scale

    def madds_total(self):
        return self.ticks * (self.scratch * self.d * self.d_key + self.ops * self.d_key +
                             self.k * (2 * self.d * self.d + self.d))


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    torch.set_num_threads(8)
    data = TASKS[task](0)
    print("%s, vvm on a vector RAM: 49 input slots, 16 scratch slots, 1 output slot, 16 numbers "
          "a slot; 2,048 vops, 32 fetched a tick; 6,000 steps" % task)
    print("| ticks | params | madds | train | test | vops used |")
    print("|---|---|---|---|---|---|")
    for ticks in (1, 2, 4, 8):
        torch.manual_seed(0)
        model = RamVVM(ticks=ticks)
        fit, test, _ = train(model, data, 6000, 128, 3e-3, 0, wd=0.05, shift=2)
        model.picks.zero_()
        score(model, data[2], data[3], "class")
        print("| %d | %d | %d | %.3f | %.3f | %d of %d |" % (
            ticks, params(model), model.madds_total(), fit, test,
            int((model.picks > 0).sum()), model.ops))


if __name__ == "__main__":
    main()
