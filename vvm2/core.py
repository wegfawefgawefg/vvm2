"""The vector virtual machine.

Memory is one list of slots, 16 numbers each, in regions:

    slot 0    the pc. Ordinary memory.
    input     written by the world, never by the program
    scratch   starts empty
    output    written by the program, read by the world

The machine has a few pointers: R read pointers and one write pointer. A pointer is a place
in memory, exactly one slot. It is not an address vector and nothing is matched to find it.
A vop moves a pointer by steps:

    stay | one of the device's own steps | go to the first or the last slot of a region
                                                                               by position
    seek: go to the slot whose contents best match a vector the vop supplies     by content

The machine has a fixed number of step kinds (`moves`, 2 unless said) and does not know
what they mean. The device says where each kind of step leads from each of its slots: on a
line, step 0 is the next slot and step 1 the previous one, stopping at the ends; on a grid a
device may make them right, left, down and up, and further ones zoom between a coarse and a
fine copy of an image. Scratch and output are lines. So a pointer is an integer and a step
is a table lookup.

A vop is a key (16 numbers) and one matrix with a bias.

One tick, the same for every task:
    fetch     the few vops whose keys are nearest the pc slot, blended by how near
    load      the slots the read pointers are on, and the slot the write pointer is on
    execute   one matmul: in, what was loaded and the pc slot; out, a value to store, the
              next pc, and for every pointer how much it wants each step
    store     the value goes to the slot under the write pointer; the pc slot takes its next
              value; the pointers take their steps

Every store uses one rule: a slot moves toward what is written by its loudness,
    slot = (1 - |c|) slot + c        with |c| at most 1
so a quiet vop nudges, a loud one replaces, a silent one does nothing.

The only softness is in the fetch. Because a few vops are blended, a pointer's step is a
blend of steps while the machine is learning, and the pointer is then spread over a few
slots; when the vops agree, the step is exact and so is the pointer. `sharp()` reports how
exact the pointers are (1.0: all on single slots).

There is no state outside memory and the pointers, no input map and no readout. Only the pc
chooses what runs. The earlier form, with address vectors matched by nearness, is
old/core6b.py; finding 32 is why it was replaced.
"""
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SLOT = 16
ANCHORS = 6         # the first and last slot of input, scratch and output


def move(slot, c, share=None):
    """The one write rule. c [.., 16] is held to length 1; share (0..1) scales the write."""
    n = c.norm(dim=-1, keepdim=True)
    c = c * (n.clamp(max=1.0) / n.clamp_min(1e-6))
    if share is not None:
        c = c * share
    return (1 - c.norm(dim=-1, keepdim=True)) * slot + c


def choose(p, draw, through):
    """p [..., n], shares that sum to 1 -> exactly one of them. draw: pick one at random by
    its share, otherwise the largest. through: the gradient passes as if the shares
    themselves had been used (straight through)."""
    flat = p.reshape(-1, p.shape[-1])
    i = torch.multinomial(flat, 1).squeeze(1) if draw else flat.argmax(dim=-1)
    hot = F.one_hot(i, p.shape[-1]).float().view_as(p)
    return hot + p - p.detach() if through else hot


class Core(nn.Module):
    def __init__(self, scratch=32, ops=128, k=4, reads=2, seek_self=False, moves=2):
        super().__init__()
        self.scratch, self.ops, self.k, self.reads = scratch, ops, k, reads
        self.moves = moves
        # a pointer's choices, in this order: stay; the device's `moves` steps; the six
        # region ends; seek; and with seek_self one more, "go to another slot like the one I
        # am on", with nothing for the vop to supply
        self.steps = 1 + moves + ANCHORS + 1 + int(seek_self)
        ptrs = reads + 1
        self.boot_pc = nn.Parameter(F.normalize(torch.randn(SLOT), dim=0) * 0.5)
        self.boot_at = nn.Parameter(torch.zeros(ptrs, 6))   # where each pointer starts
        self.keys = nn.Parameter(torch.randn(ops, SLOT) / math.sqrt(SLOT))
        fan_in = (ptrs + 1) * SLOT
        self.fan_out = 2 * SLOT + ptrs * self.steps + ptrs * SLOT   # the last part: what to seek
        self.seek_sharp = nn.Parameter(torch.tensor(12.0))
        self.m = nn.Parameter(torch.randn(ops, self.fan_out, fan_in) / math.sqrt(fan_in))
        self.bias = nn.Parameter(torch.randn(ops, self.fan_out) * 0.1)
        self.register_buffer("picks", torch.zeros(ops))
        self.trace, self.exact = None, None
        self.hard = False   # running, not training: every pointer takes its likeliest step
                            # whole, so pointers are single slots and stay so at any length
        self.one = False    # a switch under test: no blend of vops; exactly one runs a tick
        self.draw = False   # a switch under test: pointers are exact in training too. While
                            # learning, each pointer takes one move drawn by its shares and
                            # the gradient passes straight through; when running it takes
                            # the likeliest. The machine trained is then the machine run.

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
        nxt = torch.where(last, place, place + 1)           # a step stops at a region's end
        prv = torch.where(head, place, place - 1)
        to = place.repeat(self.moves, 1)                    # [moves, slots]: where each leads
        to[0] = nxt
        if self.moves > 1:
            to[1] = prv
        if go is not None:                                  # the input device's own steps
            to[:, 1:1 + n_in] = 1 + go
        anchors = F.one_hot(torch.tensor(ends), slots).float()          # [6, slots]
        can_write = torch.ones(slots)
        can_write[1:1 + n_in] = 0
        mem = torch.cat([self.boot_pc.expand(n, 1, -1), first,
                         first.new_zeros(n, self.scratch + n_out, SLOT)], dim=1)
        start = torch.softmax(self.boot_at, dim=-1).expand(n, -1, -1)
        learn = self.draw and self.training
        whole = self.draw or (self.hard and not self.training)
        if whole:
            start = choose(start, learn, learn)
        at = start @ anchors                                                 # [B, P, slots]
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
            if self.one:    # exactly one vop runs: drawn by nearness while learning (the
                mix = choose(mix, self.training, self.training)   # gradient straight
                                                                  # through), the nearest when running
            got = torch.cat([torch.einsum("bps,bsd->bpd", at, mem).flatten(1), mem[:, 0]], dim=1)
            # execute. Runs in a batch mostly fetch the same few vops, so each vop that
            # anyone fetched is applied once to the whole batch (one plain matmul) and every
            # run then takes its own k results. Same numbers as applying m[idx] run by run.
            uniq, inv = idx.unique(return_inverse=True)
            fan_in = got.shape[1]
            every = (self.m[uniq].reshape(-1, fan_in) @ got.T).view(len(uniq), self.fan_out, n)
            every = every.permute(2, 0, 1) + self.bias[uniq]                  # [B, U, fan_out]
            out = every.gather(1, inv.unsqueeze(-1).expand(-1, -1, self.fan_out))
            vec = torch.einsum("bk,bki->bi", mix, F.hardtanh(out[..., :2 * SLOT]))
            ptrs = self.reads + 1
            moves = out[..., 2 * SLOT:2 * SLOT + ptrs * self.steps]
            want = torch.softmax(moves.reshape(n, self.k, ptrs, self.steps), dim=-1)
            look = torch.einsum("bk,bki->bi", mix, out[..., 2 * SLOT + ptrs * self.steps:])
            # with exact pointers a pointer can sit on a slot that is exactly empty; an
            # empty slot then matches nothing (and its gradient stays finite)
            tiny = 1e-2 if self.draw else 1e-12
            look = F.normalize(look.view(n, ptrs, SLOT), dim=-1)
            shape = F.normalize(mem, dim=-1, eps=tiny).transpose(1, 2)
            like = look @ shape                                               # [B, P, slots]
            here = at.detach() if learn else at     # a mask, not a signal: with exact
            like = self.seek_sharp * like - 30.0 * here   # pointers its gradient compounds
            like = like.masked_fill(place == 0, float("-inf"))               # nor the pc slot
            found = torch.softmax(like, dim=-1)
            mine = F.normalize(torch.einsum("bps,bsd->bpd", at, mem), dim=-1, eps=tiny)
            twin = self.seek_sharp * (mine @ shape) - 30.0 * here
            twin = torch.softmax(twin.masked_fill(place == 0, float("-inf")), dim=-1)
            step = torch.einsum("bk,bkps->bps", mix, want)                    # [B, P, steps]
            if whole:
                step = choose(step, learn, learn)
                found = choose(found, False, learn)
                twin = choose(twin, False, learn)
            write = at[:, -1]
            mem = move(mem, vec[:, None, :SLOT], (write * can_write).unsqueeze(-1))   # store
            mem = torch.cat([move(mem[:, :1], vec[:, None, SLOT:]), mem[:, 1:]], dim=1)
            m = self.moves
            new = step[..., 0:1] * at
            for j in range(m):
                new = new + step[..., 1 + j:2 + j] * torch.zeros_like(at).index_add(2, to[j], at)
            new = new + step[..., 1 + m:1 + m + ANCHORS] @ anchors
            new = new + step[..., 1 + m + ANCHORS:2 + m + ANCHORS] * found
            if self.steps > 2 + m + ANCHORS:
                new = new + step[..., 2 + m + ANCHORS:] * twin
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
