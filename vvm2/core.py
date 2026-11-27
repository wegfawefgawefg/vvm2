"""The vector virtual machine.

One memory of slots, 16 numbers each. Every slot has an address, a vector of 16 numbers:
[ region (4) | position inside the region (12) ]. The regions, in order:

    head      the machine's own slots: 0 the pc, 1..R the read addresses, R + 1 the write
              address. Ordinary memory: anything can point at them.
    input     written by the world, never by the program
    scratch   starts empty
    output    written by the program, read by the world

The world decides how many input and output slots there are and where each one sits (its
position vector: a line, a grid, or nothing in particular). The machine does not know what
they mean.

A vop is a key (16 numbers) and one matrix with a bias.

One tick, the same for every task:
    fetch     the few vops whose keys are nearest the pc slot, blended by how near
    load      x1..xR = memory at each read address, y = memory at the write address
              (the slots whose addresses are nearest, weighted by how near)
    execute   [ v | next pc | next read addresses | next write address ]
                  = clamp(M . [x.., y, pc, read addresses, write address] + bias)
    store     memory at the write address <- v ; the head slots <- their next values

Every store uses one rule: a slot moves toward what is written by its loudness,
    slot = (1 - |c|) slot + c        with |c| at most 1
so a quiet vop nudges, a loud one replaces, a silent one does nothing, and no vector in the
machine is ever longer than 1.

There is no state outside memory, no input map and no readout. Only the pc chooses what
runs. Findings 21 to 29 in docs/findings.md are about this machine (its first form is
old/core6.py).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SLOT = 16
POS = 12
REGIONS = {"head": 0, "input": 1, "scratch": 2, "output": 3}


# Six wave rates for positions along a line. With only six, two places far apart can look
# alike: with the first set, places 12 apart are as alike as places 4 apart, and a copy loop
# on a long list writes the item 12 places away (finding 32). The second set was searched
# for the least far likeness with neighbours still distinct; it lowers the worst case from
# 0.33 to 0.25, no more: 12 numbers of position is a small address space.
LINE_RATES = {"first": [1.0, 0.5556, 0.3086, 0.1715, 0.0953, 0.0529],
              "searched": [0.856, 0.5839, 0.376, 0.1903, 0.1132, 0.0512]}
line_rates = "first"


def line_pos(n, start=0):
    """Positions along a line: slot i is sines and cosines of i at six scales, so the next
    slot is always the same small turn away, at any length."""
    i = torch.arange(start, start + n).float().unsqueeze(1)
    rate = torch.tensor(LINE_RATES[line_rates])
    return torch.cat([torch.sin(i * rate), torch.cos(i * rate)], dim=1)


def grid_pos(side):
    """Positions on a square grid: row and column, each at three scales."""
    rc = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"),
                     dim=-1).view(-1, 2).float()
    feats = []
    for axis in range(2):
        for scale in (1.0, 2.0, 4.0):
            a = rc[:, axis] * math.pi / (side * scale)
            feats += [torch.sin(a), torch.cos(a)]
    return torch.stack(feats, dim=1)


def rand_pos(n, seed):
    """Positions that mean nothing: a fixed random vector a slot."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, POS, generator=g) * 0.6


def addresses(region, pos):
    """[n, 12] positions -> [n, 16] addresses in the named region."""
    code = torch.zeros(len(pos), 4)
    code[:, REGIONS[region]] = 2.0
    return torch.cat([code, pos], dim=1)


def move(slot, c, share=None):
    """The one write rule. c [.., 16] is held to length 1; share (0..1) scales the write."""
    n = c.norm(dim=-1, keepdim=True)
    c = c * (n.clamp(max=1.0) / n.clamp_min(1e-6))
    if share is not None:
        c = c * share
    return (1 - c.norm(dim=-1, keepdim=True)) * slot + c


class Core(nn.Module):
    def __init__(self, scratch=16, ops=128, k=4, reads=2, see_head=True, sharp=8.0):
        super().__init__()
        self.scratch, self.ops, self.k, self.reads, self.see_head = scratch, ops, k, reads, see_head
        self.head = head = reads + 2
        self.register_buffer("head_where", addresses("head", rand_pos(head, 1)))
        self.register_buffer("scratch_where", addresses("scratch", rand_pos(scratch, 2)))
        self.boot = nn.Parameter(F.normalize(torch.randn(head, SLOT), dim=1) * 0.5)
        self.keys = nn.Parameter(torch.randn(ops, SLOT) / math.sqrt(SLOT))
        fan_in = (reads + 1 + (head if see_head else 0)) * SLOT
        fan_out = (1 + head) * SLOT
        self.m = nn.Parameter(torch.randn(ops, fan_out, fan_in) / math.sqrt(fan_in))
        self.bias = nn.Parameter(torch.randn(ops, fan_out) * 0.1)
        self.sharp = nn.Parameter(torch.tensor(float(sharp)))  # how sharply an address picks slots
        self.register_buffer("picks", torch.zeros(ops))
        self.trace = None

    def point(self, addr, where):
        """Address vectors [B, n, 16] -> how much each slot answers, [B, n, slots]."""
        return torch.softmax(self.sharp * (F.normalize(addr, dim=-1) @ where.T), dim=-1)

    def forward(self, world, ticks, in_pos, out_pos, trace=False):
        """Run the machine for `ticks` ticks.
        world     [B, n_in, 16], the input slots for the whole run; or a function of the
                  tick number that returns them, for input that changes while it runs
        in_pos, out_pos   [n, 12] positions of the input and output slots
        Returns the output region after every tick: [ticks, B, n_out, 16]."""
        first = world(0) if callable(world) else world
        n, n_in, n_out, head = len(first), first.shape[1], len(out_pos), self.head
        where = F.normalize(torch.cat([
            self.head_where, addresses("input", in_pos).to(first.device), self.scratch_where,
            addresses("output", out_pos).to(first.device)]), dim=1)
        can_write = torch.ones(len(where), device=first.device)
        can_write[head:head + n_in] = 0
        mem = torch.cat([self.boot.expand(n, -1, -1), first,
                         first.new_zeros(n, self.scratch + n_out, SLOT)], dim=1)
        outs, log = [], []
        for t in range(ticks):
            if callable(world) and t > 0:                 # the world rewrites its slots
                mem = torch.cat([mem[:, :head], world(t), mem[:, head + n_in:]], dim=1)
            pc = F.layer_norm(mem[:, 0], (SLOT,))
            sc, idx = (pc @ self.keys.T).topk(self.k, dim=-1)                 # fetch
            mix = torch.softmax(sc, dim=-1)
            aim = self.point(mem[:, 1:head], where)                          # [B, R + 1, slots]
            got = torch.einsum("bns,bsd->bnd", aim, mem).flatten(1)          # load
            if self.see_head:
                got = torch.cat([got, mem[:, :head].flatten(1)], dim=1)
            out = F.hardtanh(torch.einsum("bkij,bj->bki", self.m[idx], got) + self.bias[idx])
            out = torch.einsum("bk,bki->bi", mix, out).view(n, 1 + head, SLOT)   # execute
            w = aim[:, -1]
            mem = move(mem, out[:, :1], (w * can_write).unsqueeze(-1))       # store
            mem = torch.cat([move(mem[:, :head], out[:, 1:]), mem[:, head:]], dim=1)
            outs.append(mem[:, len(where) - n_out:])
            if not self.training:
                self.picks += torch.bincount(idx[:, 0], minlength=self.ops).float()
            if trace:
                log.append((idx[:, 0], aim[:, :-1].detach(), w.detach()))
        self.trace = log
        return torch.stack(outs)

    def madds(self, ticks, slots):
        fan_in = (self.reads + 1 + (self.head if self.see_head else 0)) * SLOT
        return ticks * (self.ops * SLOT + (self.reads + 1) * slots * SLOT * 2 +
                        self.k * (1 + self.head) * SLOT * fan_in)
