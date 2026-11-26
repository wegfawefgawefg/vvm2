"""The vector virtual machine (version 6): one memory, and the program computes its addresses.

  python -m vvm2.core [TASK]        default mnist; also kmnist, fashion

Memory is one list of slots, 16 numbers each. Every slot has an address: a fixed vector of
16 numbers, drawn once at random, part of the machine. The slots:
    0  pc           the machine fetches the vops whose keys are nearest this
    1  read_addr    the machine loads from wherever this points
    2  write_addr   the machine stores wherever this points
    then the world's input (49 image patches; cannot be written), scratch (starts empty)
    and the world's output (one slot; its first 10 numbers are the answer)
The pc and the two address slots are ordinary memory: anything can point at them.

A vop is a key (16 numbers) and one matrix, 64 by 32, with a bias.

One tick, the same for every task:
    fetch     the few vops whose keys are nearest the pc slot, blended by how near
    load      x = memory at read_addr, y = memory at write_addr
              (the slots whose addresses are nearest, weighted by how near)
    execute   [ v | next pc | next read_addr | next write_addr ] = clamp(M . [x, y] + bias)
    store     memory at write_addr <- v ;  slot 0 <- next pc ;  slot 1, 2 <- next addresses

Every store uses one rule: the slot moves toward what is written by its loudness,
    slot = (1 - |c|) slot + c        with |c| at most 1
so a quiet vop nudges, a loud one replaces, a silent one does nothing, and no vector in the
machine is ever longer than 1.

A vop sees only the two vectors that were loaded. Its bias is what it does "by default":
where to go next and where to point next, whatever the data. There is no state outside
memory, no input map, no readout, and nothing but the pc chooses what runs.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classic import TASKS, params, score, train
from .ram import patches

SLOT = 16


def move(slot, c, share=None):
    """The one write rule. c [.., 16] is held to length 1; share (0..1) scales the write."""
    n = c.norm(dim=-1, keepdim=True)
    c = c * (n.clamp(max=1.0) / n.clamp_min(1e-6))
    if share is not None:
        c = c * share
    return (1 - c.norm(dim=-1, keepdim=True)) * slot + c


def grid_addresses(n_in, scratch, g):
    """Structured addresses: [ region (4 numbers) | position inside it (12 numbers) ].
    Region: one of four on/off codes (the machine's own slots, the input, scratch, the
    output). Position: for the input, the patch's row and column written smoothly (sines and
    cosines at three scales, 6 numbers each), so a neighbouring patch has a nearby address
    and "one to the right" is the same small step everywhere. Elsewhere, a random vector."""
    side = math.isqrt(n_in)
    slots = 3 + n_in + scratch + 1
    where = torch.zeros(slots, SLOT)
    region = torch.eye(4) * 2.0
    where[:3, :4], where[3:3 + n_in, :4] = region[0], region[1]
    where[3 + n_in:-1, :4], where[-1, :4] = region[2], region[3]
    where[:, 4:] = torch.randn(slots, 12, generator=g) * 0.5
    rc = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"),
                     dim=-1).view(-1, 2).float()
    feats = []
    for axis in range(2):
        for scale in (1.0, 2.0, 4.0):
            a = rc[:, axis] * math.pi / (side * scale)
            feats += [torch.sin(a), torch.cos(a)]
    where[3:3 + n_in, 4:] = torch.stack(feats, dim=1)
    return where


class Core(nn.Module):
    def __init__(self, n_in=49, scratch=16, ops=256, k=4, ticks=16, addresses="random"):
        super().__init__()
        self.n_in, self.scratch, self.ops, self.k, self.ticks = n_in, scratch, ops, k, ticks
        slots = 3 + n_in + scratch + 1
        g = torch.Generator().manual_seed(1)
        where = torch.randn(slots, SLOT, generator=g)
        if addresses == "grid":
            where = grid_addresses(n_in, scratch, g)
        self.register_buffer("where", F.normalize(where, dim=1))
        can = torch.ones(slots)
        can[3:3 + n_in] = 0                               # the world's input cannot be written
        self.register_buffer("can_write", can)
        self.boot = nn.Parameter(F.normalize(torch.randn(3, SLOT), dim=1) * 0.5)  # reset values
        self.keys = nn.Parameter(torch.randn(ops, SLOT) / math.sqrt(SLOT))
        self.m = nn.Parameter(torch.randn(ops, 4 * SLOT, 2 * SLOT) / math.sqrt(2 * SLOT))
        self.bias = nn.Parameter(torch.randn(ops, 4 * SLOT) * 0.1)
        self.sharp = nn.Parameter(torch.tensor(8.0))      # how sharply an address picks slots
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.register_buffer("picks", torch.zeros(ops))
        self.trace = None

    def point(self, addr):
        """Address vector [B, 16] -> how much each slot answers, [B, slots]."""
        return torch.softmax(self.sharp * (F.normalize(addr, dim=1) @ self.where.T), dim=-1)

    def forward(self, x, trace=False):
        n = len(x)
        mem = torch.cat([self.boot.expand(n, -1, -1), patches(x),
                         x.new_zeros(n, self.scratch + 1, SLOT)], dim=1)
        log = []
        for _ in range(self.ticks):
            pc = F.layer_norm(mem[:, 0], (SLOT,))
            sc, idx = (pc @ self.keys.T).topk(self.k, dim=-1)                 # fetch
            mix = torch.softmax(sc, dim=-1)
            r, w = self.point(mem[:, 1]), self.point(mem[:, 2])              # [B, slots]
            got = torch.cat([torch.einsum("bs,bsd->bd", r, mem),
                             torch.einsum("bs,bsd->bd", w, mem)], dim=1)     # load x, y
            out = F.hardtanh(torch.einsum("bkij,bj->bki", self.m[idx], got) + self.bias[idx])
            out = torch.einsum("bk,bki->bi", mix, out)                       # execute
            v, npc, nra, nwa = out.split(SLOT, dim=1)
            share = (w * self.can_write).unsqueeze(-1)                       # store
            mem = move(mem, v.unsqueeze(1), share)
            head = torch.stack([move(mem[:, 0], npc), move(mem[:, 1], nra),
                                move(mem[:, 2], nwa)], dim=1)
            mem = torch.cat([head, mem[:, 3:]], dim=1)
            if not self.training:
                self.picks += torch.bincount(idx[:, 0], minlength=self.ops).float()
            if trace:
                log.append((idx[:, 0], r.argmax(dim=1), w.argmax(dim=1), v.norm(dim=1)))
        self.trace = log
        return mem[:, -1, :10] * self.scale

    def madds_total(self):
        slots = self.where.shape[0]
        return self.ticks * (self.ops * SLOT + 2 * slots * SLOT * 2 + self.k * 4 * SLOT * 2 * SLOT
                             + 4 * SLOT * 2)


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    torch.set_num_threads(8)
    data = TASKS[task](0)
    print("%s, the vector virtual machine v6: one memory (pc, 2 address slots, 49 input, 16 "
          "scratch, 1 output), 256 vops of 64x32, 4 blended a tick; 6,000 steps" % task)
    print("| ticks | params | madds | train | test | vops run | slots read a run | slots "
          "written a run |", flush=True)
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for ticks in (8, 16, 32):
        torch.manual_seed(0)
        model = Core(ticks=ticks)
        fit, test, _ = train(model, data, 6000, 128, 3e-3, 0, wd=0.05, shift=2)
        model.picks.zero_()
        score(model, data[2], data[3], "class")
        model.eval()
        with torch.no_grad():
            model(data[2][:2000], trace=True)

        def distinct(j):
            seq = torch.stack([t[j] for t in model.trace], dim=1)
            return torch.tensor([len(set(r.tolist())) for r in seq]).float().mean().item()

        print("| %d | %d | %d | %.3f | %.3f | %d of %d | %.1f | %.1f |" % (
            ticks, params(model), model.madds_total(), fit, test,
            int((model.picks > 0).sum()), model.ops, distinct(1), distinct(2)), flush=True)


if __name__ == "__main__":
    main()
