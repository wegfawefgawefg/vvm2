"""The vector virtual machine.

Memory is one list of slots, 16 numbers each:

    slot 0    the pc. Ordinary memory.
    input     written by the world, never by the program
    scratch   starts empty
    output    written by the program, read by the world

The machine has a few pointers: R read pointers and one write pointer. A pointer is a place
in memory, exactly one slot. A vop moves a pointer in one way only, by position:

    stay | one of the device's own steps

The machine has a fixed number of step kinds (`moves`, 2 unless said) and does not know what
they mean. The device says where each kind of step leads from each of its slots: on a line,
step 0 is the next slot and step 1 the previous one, stopping at the ends; a device may make
them a ring, or right, left, down and up on a grid. Scratch and output are lines.

There is no move by content and no jump to a region's end (both were in the form before
this one, old/core7.py; findings 43 and 45 are why they went). The only thing found by
content is the vop: the fetch.

A vop is a key (16 numbers) and one matrix with a bias.

One tick, the same for every task:
    fetch     the few vops whose keys are nearest the pc slot, blended by how near
    load      the slots the read pointers are on, and the slot the write pointer is on
    execute   one matmul: in, what was loaded and the pc slot; out, a value to store, the
              next pc, and for every pointer how much it wants each move
    store     the value goes to the slot under the write pointer; the pc slot takes its next
              value; the pointers move

Every store uses one rule: a slot moves toward what is written by its loudness,
    slot = (1 - |c|) slot + c        with |c| at most 1
so a quiet vop nudges, a loud one replaces, a silent one does nothing.

Where a pointer starts is learned: one of the six region ends (the first or last slot of
input, scratch or output), the same for every run. That and "the program cannot write the
input" are all the machine knows about regions.

Pointers while learning. As first built, a pointer's move is a blend of moves while the
machine learns (because the few fetched vops are blended), so the pointer is spread over
several slots, and it is made exact only when running (`hard = True`). With `draw = True`
the pointer is exact while learning too: it takes its likeliest move outright and the
gradient is passed straight through, so the machine trained is the machine run (finding 46).
"""
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SLOT = 16


def move(slot, c, share=None):
    """The one write rule. c [.., 16] is held to length 1; share (0..1) scales the write."""
    n = c.norm(dim=-1, keepdim=True)
    c = c * (n.clamp(max=1.0) / n.clamp_min(1e-6))
    if share is not None:
        c = c * share
    return (1 - c.norm(dim=-1, keepdim=True)) * slot + c


def choose(p, through):
    """p [..., n], shares that sum to 1 -> the largest one outright. through: the gradient
    passes as if the shares themselves had been used (straight through)."""
    hot = F.one_hot(p.argmax(dim=-1), p.shape[-1]).float()
    return hot + p - p.detach() if through else hot


class Core(nn.Module):
    def __init__(self, scratch=32, ops=32, k=4, reads=2, moves=2):
        super().__init__()
        self.scratch, self.ops, self.k, self.reads, self.moves = scratch, ops, k, reads, moves
        self.steps = 1 + moves          # a pointer's choices: stay, then the device's steps
        ptrs = reads + 1
        self.boot_pc = nn.Parameter(F.normalize(torch.randn(SLOT), dim=0) * 0.5)
        self.boot_at = nn.Parameter(torch.zeros(ptrs, 6))   # where each pointer starts
        self.keys = nn.Parameter(torch.randn(ops, SLOT) / math.sqrt(SLOT))
        fan_in = (ptrs + 1) * SLOT
        self.fan_out = 2 * SLOT + ptrs * self.steps
        self.m = nn.Parameter(torch.randn(ops, self.fan_out, fan_in) / math.sqrt(fan_in))
        self.bias = nn.Parameter(torch.randn(ops, self.fan_out) * 0.1)
        self.register_buffer("picks", torch.zeros(ops))
        self.trace, self.exact = None, None
        self.hard = False   # running, not training: every pointer takes its likeliest move
        self.draw = False   # pointers exact while learning too (see the top of the file)

    def forward(self, world, ticks, n_out, trace=False, go=None):
        """Run for `ticks` ticks.
        world     [B, n_in, 16], the input slots for the whole run; or a function that
                  returns them tick by tick, for input that changes while it runs:
                  world(t), or world(t, out) to be handed the output slots as they stand
                  (None before the first tick), so the world can answer what the machine
                  does. If it returns None the run ends there.
        n_out     how many output slots the world wants
        go        the device's steps: a whole-number table [moves, n_in]; go[m, i] is the
                  input slot that step m leads to from input slot i. Left out: a line.
        Returns the output region after every tick: [ticks, B, n_out, 16]."""
        hands = callable(world) and len(inspect.signature(world).parameters) > 1
        first = (world(0, None) if hands else world(0)) if callable(world) else world
        n, n_in = len(first), first.shape[1]
        slots = 1 + n_in + self.scratch + n_out
        ends = [1, n_in, 1 + n_in, n_in + self.scratch, 1 + n_in + self.scratch, slots - 1]
        place = torch.arange(slots)
        last = torch.zeros(slots, dtype=torch.bool)
        last[[0, ends[1], ends[3], ends[5]]] = True
        head = torch.zeros(slots, dtype=torch.bool)
        head[[0, ends[0], ends[2], ends[4]]] = True
        to = place.repeat(self.moves, 1)                    # [moves, slots]: where each leads
        to[0] = torch.where(last, place, place + 1)         # a step stops at a region's end
        if self.moves > 1:
            to[1] = torch.where(head, place, place - 1)
        if go is not None:                                  # the input device's own steps
            to[:, 1:1 + n_in] = 1 + go
        can_write = torch.ones(slots)
        can_write[1:1 + n_in] = 0
        mem = torch.cat([self.boot_pc.expand(n, 1, -1), first,
                         first.new_zeros(n, self.scratch + n_out, SLOT)], dim=1)
        learn = self.draw and self.training
        whole = self.draw or (self.hard and not self.training)
        start = torch.softmax(self.boot_at, dim=-1).expand(n, -1, -1)
        if whole:
            start = choose(start, learn)
        at = start @ F.one_hot(torch.tensor(ends), slots).float()            # [B, P, slots]
        ptrs, fan_in = self.reads + 1, (self.reads + 2) * SLOT
        outs, log, exact = [], [], []
        for t in range(ticks):
            if callable(world) and t > 0:                 # the world rewrites its slots
                now = world(t, mem[:, slots - n_out:]) if hands else world(t)
                if now is None:
                    break
                mem = torch.cat([mem[:, :1], now, mem[:, 1 + n_in:]], dim=1)
            pc = F.layer_norm(mem[:, 0], (SLOT,))
            sc, idx = (pc @ self.keys.T).topk(self.k, dim=-1)                 # fetch
            mix = torch.softmax(sc, dim=-1)
            got = torch.cat([torch.einsum("bps,bsd->bpd", at, mem).flatten(1), mem[:, 0]], dim=1)
            # execute. Runs in a batch mostly fetch the same few vops, so each vop that
            # anyone fetched is applied once to the whole batch (one plain matmul) and every
            # run then takes its own k results.
            uniq, inv = idx.unique(return_inverse=True)
            every = (self.m[uniq].reshape(-1, fan_in) @ got.T).view(len(uniq), self.fan_out, n)
            every = every.permute(2, 0, 1) + self.bias[uniq]                  # [B, U, fan_out]
            out = every.gather(1, inv.unsqueeze(-1).expand(-1, -1, self.fan_out))
            vec = torch.einsum("bk,bki->bi", mix, F.hardtanh(out[..., :2 * SLOT]))
            want = torch.softmax(out[..., 2 * SLOT:].reshape(n, self.k, ptrs, self.steps), dim=-1)
            step = torch.einsum("bk,bkps->bps", mix, want)                    # [B, P, steps]
            if whole:
                step = choose(step, learn)
            mem = move(mem, vec[:, None, :SLOT], (at[:, -1] * can_write).unsqueeze(-1))   # store
            mem = torch.cat([move(mem[:, :1], vec[:, None, SLOT:]), mem[:, 1:]], dim=1)
            new = step[..., 0:1] * at
            for j in range(self.moves):
                new = new + step[..., 1 + j:2 + j] * torch.zeros_like(at).index_add(2, to[j], at)
            at = new
            outs.append(mem[:, slots - n_out:])
            exact.append(at.max(dim=-1).values.mean().detach())
            if not self.training:
                self.picks += torch.bincount(idx[:, 0], minlength=self.ops).float()
            if trace:
                log.append((idx[:, 0], at.detach(), step.detach()))
        self.trace, self.exact = log, torch.stack(exact).mean().item()
        return torch.stack(outs)

    def sharp(self):
        """How exact the pointers were on the last run: the mean share of a pointer that is
        on its single likeliest slot (1.0: every pointer on one slot at every tick)."""
        return self.exact
