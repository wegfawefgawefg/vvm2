"""The vector virtual machine, minimal form.

  python -m vvm2.vm [TASK]        default mnist; also kmnist, fashion

Memory: slots of 16 numbers. Every slot has a fixed address (a vector of 16 numbers, drawn
once at random; it is part of the machine, not learned). For an image task the slots are 49
input patches (4 by 4 pixels, cannot be written), some scratch slots (start empty) and one
output slot whose first 10 numbers are the answer.

State = [ pc | addr | data ]: which vop is next, which memory it is working on, and a working
value. It starts from a learned reset value, the same for every run.

One tick, the same for every task:
    fetch     the few vops whose keys are nearest the pc lane, blended by how near
    load      x = memory read at the addr lane (the slots whose addresses are nearest,
              weighted by how near)
    execute   state, v = clamp(vop . [state, x])      one matrix; it writes the next pc and
              the next addr along with everything else
    store     the slots just read move toward v by v's loudness:
              slot = (1 - w |v|) slot + w v           w: how near the slot's address was

Nothing else computes. There is no input map and no readout.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classic import TASKS, params, score, train
from .ram import patches

SLOT, PC, ADDR = 16, 16, 16


class VM(nn.Module):
    def __init__(self, n_in=49, scratch=16, data=64, ops=64, k=4, ticks=16, two=False):
        super().__init__()
        self.n_in, self.scratch, self.data, self.ops, self.k, self.ticks = n_in, scratch, data, ops, k, ticks
        self.two = two  # True: a read address and a separate write address
        self.lanes = PC + ADDR * (2 if two else 1)
        self.width = self.lanes + data
        slots = n_in + scratch + 1
        g = torch.Generator().manual_seed(1)
        self.register_buffer("where", F.normalize(torch.randn(slots, ADDR, generator=g), dim=1))
        can = torch.ones(slots)
        can[:n_in] = 0                                    # input slots cannot be written
        self.register_buffer("can_write", can)
        self.reset = nn.Parameter(torch.randn(self.width) * 0.1)
        self.keys = nn.Parameter(torch.randn(ops, PC) / math.sqrt(PC))
        fan = self.width + SLOT
        self.m = nn.Parameter(torch.randn(ops, fan, fan) / math.sqrt(fan))
        self.bias = nn.Parameter(torch.zeros(ops, fan))
        self.sharp = nn.Parameter(torch.tensor(8.0))     # how sharply an address picks slots
        self.scale = nn.Parameter(torch.tensor(3.0))
        self.register_buffer("picks", torch.zeros(ops))
        self.trace = None

    def forward(self, x, trace=False):
        n = len(x)
        mem = torch.cat([patches(x), x.new_zeros(n, self.scratch + 1, SLOT)], dim=1)
        state = self.reset.expand(n, -1)
        log = []
        for _ in range(self.ticks):
            pc = F.layer_norm(state[:, :PC], (PC,))
            sc, idx = (pc @ self.keys.T).topk(self.k, dim=-1)                # fetch
            mix = torch.softmax(sc, dim=-1)
            addr = F.normalize(state[:, PC:PC + ADDR], dim=1)
            w = torch.softmax(self.sharp * (addr @ self.where.T), dim=-1)    # [B, slots]
            to = w
            if self.two:
                waddr = F.normalize(state[:, PC + ADDR:PC + 2 * ADDR], dim=1)
                to = torch.softmax(self.sharp * (waddr @ self.where.T), dim=-1)
            got = torch.einsum("bs,bsd->bd", w, mem)                         # load
            seen = torch.cat([state, got], dim=1)
            out = torch.einsum("bkij,bj->bki", self.m[idx], seen) + self.bias[idx]
            out = torch.einsum("bk,bki->bi", mix, F.hardtanh(out))           # execute
            state, v = out[:, :self.width], out[:, self.width:]
            loud = v.norm(dim=1, keepdim=True).clamp(max=1.0)
            v = v * (loud / v.norm(dim=1, keepdim=True).clamp_min(1e-6))
            share = (to * self.can_write).unsqueeze(-1)                      # store
            mem = (1 - share * loud.unsqueeze(1)) * mem + share * v.unsqueeze(1)
            if not self.training:
                self.picks += torch.bincount(idx[:, 0], minlength=self.ops).float()
            if trace:
                log.append((idx[:, 0], w.argmax(dim=1), loud.squeeze(1)))
        self.trace = log
        return mem[:, -1, :10] * self.scale

    def madds_total(self):
        fan = self.width + SLOT
        slots = self.n_in + self.scratch + 1
        return self.ticks * (self.ops * PC + slots * ADDR + slots * SLOT * 2 + self.k * fan * fan)


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    torch.set_num_threads(8)
    data = TASKS[task](0)
    print("%s, the vector virtual machine: 49 input, 16 scratch, 1 output slot; state pc 16, "
          "addr 16, data 64; 64 vops, 4 blended a tick; 6,000 steps" % task)
    print("| addresses | ticks | params | madds | train | test | vops run (best match) | slots it points at |", flush=True)
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for two, ticks in ((False, 4), (True, 4), (False, 8), (True, 8), (False, 16), (True, 16)):
        torch.manual_seed(0)
        model = VM(ticks=ticks, two=two)
        fit, test, _ = train(model, data, 6000, 128, 3e-3, 0, wd=0.05, shift=2)
        model.picks.zero_()
        score(model, data[2], data[3], "class")
        model.eval()
        with torch.no_grad():
            model(data[2][:2000], trace=True)
        seen = torch.stack([t[1] for t in model.trace], dim=1)          # [n, ticks]
        distinct = torch.tensor([len(set(r.tolist())) for r in seen]).float().mean()
        print("| %s | %d | %d | %d | %.3f | %.3f | %d of %d | %.1f different a run |" % (
            "read and write" if two else "one", ticks, params(model), model.madds_total(), fit, test,
            int((model.picks > 0).sum()), model.ops, distinct), flush=True)


if __name__ == "__main__":
    main()
