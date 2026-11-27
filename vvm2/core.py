"""The vector virtual machine.

Memory is one list of slots, 16 numbers each, in regions:

    slot 0    the pc. Ordinary memory.
    input     written by the world, never by the program
    scratch   starts empty
    output    written by the program, read by the world

The machine has a few pointers: R read pointers and one write pointer. A pointer is a place
in memory, exactly one slot. It is not an address vector and nothing is matched to find it.
A vop moves a pointer by steps:

    stay | next | previous | go to the first or the last slot of a region      by position
    seek: go to the slot whose contents best match a vector the vop supplies     by content

"next" and "previous" mean the neighbouring slot of the same region and stop at its ends.
(A device with another shape would say what its own steps are: for an image, up, down,
left and right.) So a pointer is an integer and a step is adding one.

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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SLOT = 16
STEPS = 3 + 6 + 1   # stay, next, previous; the first and last slot of input, scratch and
                    # output; seek (by content)


def move(slot, c, share=None):
    """The one write rule. c [.., 16] is held to length 1; share (0..1) scales the write."""
    n = c.norm(dim=-1, keepdim=True)
    c = c * (n.clamp(max=1.0) / n.clamp_min(1e-6))
    if share is not None:
        c = c * share
    return (1 - c.norm(dim=-1, keepdim=True)) * slot + c


class Core(nn.Module):
    def __init__(self, scratch=32, ops=128, k=4, reads=2):
        super().__init__()
        self.scratch, self.ops, self.k, self.reads = scratch, ops, k, reads
        ptrs = reads + 1
        self.boot_pc = nn.Parameter(F.normalize(torch.randn(SLOT), dim=0) * 0.5)
        self.boot_at = nn.Parameter(torch.zeros(ptrs, 6))   # where each pointer starts
        self.keys = nn.Parameter(torch.randn(ops, SLOT) / math.sqrt(SLOT))
        fan_in = (ptrs + 1) * SLOT
        self.fan_out = 2 * SLOT + ptrs * STEPS + ptrs * SLOT   # the last part: what to seek
        self.seek_sharp = nn.Parameter(torch.tensor(12.0))
        self.m = nn.Parameter(torch.randn(ops, self.fan_out, fan_in) / math.sqrt(fan_in))
        self.bias = nn.Parameter(torch.randn(ops, self.fan_out) * 0.1)
        self.register_buffer("picks", torch.zeros(ops))
        self.trace, self.exact = None, None
        self.hard = False   # running, not training: every pointer takes its likeliest step
                            # whole, so pointers are single slots and stay so at any length

    def forward(self, world, ticks, n_out, trace=False):
        """Run for `ticks` ticks.
        world     [B, n_in, 16], the input slots for the whole run; or a function of the
                  tick number that returns them, for input that changes while it runs
        n_out     how many output slots the world wants
        Returns the output region after every tick: [ticks, B, n_out, 16]."""
        first = world(0) if callable(world) else world
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
        anchors = F.one_hot(torch.tensor(ends), slots).float()          # [6, slots]
        can_write = torch.ones(slots)
        can_write[1:1 + n_in] = 0
        mem = torch.cat([self.boot_pc.expand(n, 1, -1), first,
                         first.new_zeros(n, self.scratch + n_out, SLOT)], dim=1)
        start = torch.softmax(self.boot_at, dim=-1)
        if self.hard and not self.training:
            start = F.one_hot(start.argmax(dim=-1), 6).float()
        at = (start @ anchors).expand(n, -1, -1)                             # [B, P, slots]
        outs, log, exact = [], [], []
        for t in range(ticks):
            if callable(world) and t > 0:                 # the world rewrites its slots
                mem = torch.cat([mem[:, :1], world(t), mem[:, 1 + n_in:]], dim=1)
            pc = F.layer_norm(mem[:, 0], (SLOT,))
            sc, idx = (pc @ self.keys.T).topk(self.k, dim=-1)                 # fetch
            mix = torch.softmax(sc, dim=-1)
            got = torch.cat([torch.einsum("bps,bsd->bpd", at, mem).flatten(1), mem[:, 0]], dim=1)
            out = torch.einsum("bkij,bj->bki", self.m[idx], got) + self.bias[idx]   # execute
            vec = torch.einsum("bk,bki->bi", mix, F.hardtanh(out[..., :2 * SLOT]))
            ptrs = self.reads + 1
            moves = out[..., 2 * SLOT:2 * SLOT + ptrs * STEPS]
            want = torch.softmax(moves.reshape(n, self.k, ptrs, STEPS), dim=-1)
            look = torch.einsum("bk,bki->bi", mix, out[..., 2 * SLOT + ptrs * STEPS:])
            look = F.normalize(look.view(n, ptrs, SLOT), dim=-1)
            like = look @ F.normalize(mem, dim=-1).transpose(1, 2)            # [B, P, slots]
            like = self.seek_sharp * like - 30.0 * at                        # not where it is
            like = like.masked_fill(place == 0, float("-inf"))               # nor the pc slot
            found = torch.softmax(like, dim=-1)
            step = torch.einsum("bk,bkps->bps", mix, want)                    # [B, P, STEPS]
            if self.hard and not self.training:
                step = F.one_hot(step.argmax(dim=-1), STEPS).float()
                found = F.one_hot(found.argmax(dim=-1), slots).float()
            write = at[:, -1]
            mem = move(mem, vec[:, None, :SLOT], (write * can_write).unsqueeze(-1))   # store
            mem = torch.cat([move(mem[:, :1], vec[:, None, SLOT:]), mem[:, 1:]], dim=1)
            fwd = torch.zeros_like(at).index_add(2, nxt, at)
            back = torch.zeros_like(at).index_add(2, prv, at)
            at = (step[..., 0:1] * at + step[..., 1:2] * fwd + step[..., 2:3] * back
                  + step[..., 3:9] @ anchors + step[..., 9:10] * found)
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
