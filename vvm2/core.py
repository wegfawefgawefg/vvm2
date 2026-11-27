"""The vector virtual machine: one memory, small vops, several cores.

Memory is one list of slots, 16 numbers each. Everything is in it:

    registers   two slots a core: its pc and its control slot
    input       written by the world, never by the program
    output      written by the program, read by the world
    scratch     starts empty
    vops        each a label slot followed by 16 slots, which are a 16 by 16 matrix

A core is a pc (a slot), a read pointer and a write pointer (places: one slot each).
Every tick, every core, all at once:

    fetch     find the slots in memory most like its pc (any slot but a register can
              match); the 16 slots after a match are a matrix. The few best are blended.
    load      the slot under its read pointer
    execute   one matmul, 16 by 16: a slot in, a slot out, clamped
    store     the result goes to the slot under its write pointer

Every store uses one rule: a slot moves toward what is written by its loudness,
    slot = (1 - |c|) slot + c        with |c| at most 1
so a quiet result nudges, a loud one replaces, a silent one does nothing.

A vop makes one slot and nothing else. It does not say where the pointers go or what runs
next. Those are memory too: a core's pc is a slot, and its control slot holds two numbers,
how far to step the read pointer and how far the write pointer this tick (+1 the next
slot, -1 the previous one, 0 stay; between, part of the way). Any core can store into any
register, its own or another's, exactly as into any other slot. A register keeps its value
until something writes it, so a core left alone runs the same vop again with its pointers
stepping the same way: a loop is the default.

Vops are memory, so a core can write one, and a pc can land on something written a moment
ago. What is learned is what memory holds when a run starts: the vops, each core's pc and
control slot, and where each pointer starts (one of a few named slots).

Memory is one line: a step goes to the neighbouring slot, whatever is there, and stops only
at the two ends of memory. A device may give its own steps for the input slots (`go`).

While learning, a pointer may be part-way between slots and the fetch is a blend. When
running (`hard = True`) every pointer step is a whole one, so a pointer is one slot.

The forms before this one are in old/ (core8: one vop doing everything, a bank apart from
memory).
"""
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SLOT = 16
ROWS = 16     # a vop: one label slot, then this many slots, the rows of its matrix


def move(slot, c, share=None):
    """The one write rule. c [.., 16] is held to length 1; share (0..1) scales the write."""
    n = c.norm(dim=-1, keepdim=True)
    c = c * (n.clamp(max=1.0) / n.clamp_min(1e-6))
    if share is not None:
        c = c * share
    return (1 - c.norm(dim=-1, keepdim=True)) * slot + c


class Core(nn.Module):
    def __init__(self, cores=4, vops=32, scratch=32, k=4):
        super().__init__()
        self.cores, self.vops, self.scratch, self.k = cores, vops, scratch, k
        self.boot_pc = nn.Parameter(torch.randn(cores, SLOT) / math.sqrt(SLOT))
        self.boot_ctl = nn.Parameter(torch.zeros(cores, SLOT))
        self.bank = nn.Parameter(torch.randn(vops, 1 + ROWS, SLOT) / math.sqrt(SLOT))
        self.homes = 5 + 2 * cores     # the named slots a pointer can start on
        self.boot_at = nn.Parameter(torch.zeros(cores, 2, self.homes))
        self.trace, self.exact = None, None
        self.hard = False              # running, not learning: whole steps only

    def layout(self, n_in, n_out):
        """Where things are: the first slot of input, output, scratch, vops, and the size."""
        reg = 2 * self.cores
        inp, out = reg, reg + n_in
        scr = out + n_out
        vop = scr + self.scratch
        return inp, out, scr, vop, vop + self.vops * (1 + ROWS)

    def forward(self, world, ticks, n_out, trace=False, go=None):
        """Run for `ticks` ticks.
        world     [B, n_in, 16], the input slots for the whole run; or a function that
                  returns them tick by tick: world(t), or world(t, out) to be handed the
                  output slots as they stand (None before the first tick). If it returns
                  None the run ends there.
        n_out     how many output slots the world wants
        go        the input device's own steps: [2, n_in] whole numbers; go[0, i] is the
                  input slot "+1" leads to from input slot i, go[1, i] the one "-1" leads to
        Returns the output slots after every tick: [ticks, B, n_out, 16]."""
        hands = callable(world) and len(inspect.signature(world).parameters) > 1
        first = (world(0, None) if hands else world(0)) if callable(world) else world
        n, n_in, C = len(first), first.shape[1], self.cores
        inp, out, scr, vop, N = self.layout(n_in, n_out)
        place = torch.arange(N)
        fwd, back = (place + 1).clamp(max=N - 1), (place - 1).clamp(min=0)
        if go is not None:
            fwd[inp:out], back[inp:out] = inp + go[0], inp + go[1]
        can_write = torch.ones(N)
        can_write[inp:out] = 0
        no_fetch = torch.zeros(N, dtype=torch.bool)       # a register is not a vop, and the
        no_fetch[:2 * C] = True                           # last slots have no 16 after them
        no_fetch[N - ROWS:] = True
        regs = torch.stack([self.boot_pc, self.boot_ctl], dim=1).reshape(2 * C, SLOT)
        mem = torch.cat([regs.expand(n, -1, -1), first, first.new_zeros(n, n_out + self.scratch, SLOT),
                         self.bank.reshape(-1, SLOT).expand(n, -1, -1)], dim=1)
        homes = torch.tensor([inp, out - 1, out, scr - 1, scr] + list(range(2 * C)))
        start = torch.softmax(self.boot_at, dim=-1)                          # [C, 2, homes]
        if self.hard and not self.training:
            start = F.one_hot(start.argmax(dim=-1), self.homes).float()
        at = (start @ F.one_hot(homes, N).float()).expand(n, -1, -1, -1)     # [B, C, 2, N]
        rows = 1 + torch.arange(ROWS)
        batch = torch.arange(n)[:, None, None, None]
        outs, log, exact = [], [], []
        for t in range(ticks):
            if callable(world) and t > 0:                 # the world rewrites its slots
                now = world(t, mem[:, out:scr]) if hands else world(t)
                if now is None:
                    break
                mem = torch.cat([mem[:, :inp], now, mem[:, out:]], dim=1)
            pc = F.layer_norm(mem[:, 0:2 * C:2], (SLOT,))                    # [B, C, 16]
            like = (pc @ mem.transpose(1, 2)).masked_fill(no_fetch, float("-inf"))
            sc, j = like.topk(self.k, dim=-1)                                 # fetch
            mix = torch.softmax(sc, dim=-1)                                   # [B, C, k]
            blocks = mem[batch, j.unsqueeze(-1) + rows]                       # [B, C, k, 16, 16]
            m = torch.einsum("bck,bckrd->bcrd", mix, blocks)
            x = torch.einsum("bcn,bnd->bcd", at[:, :, 0], mem)                # load
            y = F.hardtanh(torch.einsum("bcrd,bcd->bcr", m, x))               # execute
            for c in range(C):                                                # store
                mem = move(mem, y[:, c, None], (at[:, c, 1] * can_write).unsqueeze(-1))
            step = mem[:, 1:2 * C:2, :2].clamp(-1, 1)                         # [B, C, 2]
            if self.hard and not self.training:
                step = step.round()
            s = step.unsqueeze(-1)
            at = ((1 - s.abs()) * at + s.clamp(min=0) * torch.zeros_like(at).index_add(3, fwd, at)
                  + (-s).clamp(min=0) * torch.zeros_like(at).index_add(3, back, at))
            outs.append(mem[:, out:scr])
            exact.append(at.max(dim=-1).values.mean().detach())
            if trace:
                log.append((j[:, :, 0], at.detach(), step.detach()))
        self.trace, self.exact = log, torch.stack(exact).mean().item()
        return torch.stack(outs)

    def name(self, i, n_in, n_out):
        """A slot number as a name: pc0, ct0 (core 0's registers), i3, o2, s0, v5 (the label
        of vop 5), v5.3 (the third row of its matrix)."""
        inp, out, scr, vop, _ = self.layout(n_in, n_out)
        if i < inp:
            return ("pc%d" if i % 2 == 0 else "ct%d") % (i // 2)
        if i < out:
            return "i%d" % (i - inp)
        if i < scr:
            return "o%d" % (i - out)
        if i < vop:
            return "s%d" % (i - scr)
        v, r = divmod(i - vop, 1 + ROWS)
        return "v%d" % v if r == 0 else "v%d.%d" % (v, r)

    def sharp(self):
        """How exact the pointers were on the last run: the mean share of a pointer that is
        on its single likeliest slot (1.0: every pointer on one slot at every tick)."""
        return self.exact
