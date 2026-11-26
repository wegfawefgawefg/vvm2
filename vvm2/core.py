"""The vector virtual machine (version 6): one memory, and the program computes its addresses.

  python -m vvm2.core [TASK]        default mnist; also kmnist, fashion
  python -m vvm2.core checks        what is holding the score down: one change at a time
  python -m vvm2.core explore       is it an exploration problem: more steps, address noise
  python -m vvm2.core anneal        address noise on a curriculum: strong first, fading out
  python -m vvm2.core best          the changes that helped, together, trained longer

Memory is one list of slots, 16 numbers each. Every slot has an address: a fixed vector of
16 numbers, part of the machine. The slots:
    0            pc           the machine fetches the vops whose keys are nearest this
    1 .. R       read_addr    the machine loads from wherever each of these points
    R + 1        write_addr   the machine stores wherever this points
    then the world's input (49 image patches; cannot be written), scratch (starts empty)
    and the world's output (one slot; its first 10 numbers are the answer)
The pc and the address slots are ordinary memory: anything can point at them.

A vop is a key (16 numbers) and one matrix with a bias.

One tick, the same for every task:
    fetch     the few vops whose keys are nearest the pc slot, blended by how near
    load      x1..xR = memory at each read_addr, y = memory at write_addr
              (the slots whose addresses are nearest, weighted by how near)
    execute   [ v | next pc | next read_addrs | next write_addr ] = clamp(M . [x.., y] + bias)
    store     memory at write_addr <- v ;  the pc and address slots <- their next values

Every store uses one rule: the slot moves toward what is written by its loudness,
    slot = (1 - |c|) slot + c        with |c| at most 1
so a quiet vop nudges, a loud one replaces, a silent one does nothing, and no vector in the
machine is ever longer than 1.

A vop sees only the vectors that were loaded (and the pc slot, with see_pc). Its bias is what
it does "by default". There is no state outside memory, no input map, no readout, and
nothing but the pc chooses what runs.

Switches: reads (R, default 1), see_pc, addresses ("random" or "grid": region and smooth
row and column), top (>0: an address reads and writes only its `top` nearest slots),
jitter (noise on every address while training).
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


def grid_addresses(head, n_in, scratch, g):
    """Structured addresses: [ region (4 numbers) | position inside it (12 numbers) ].
    Region: one of four on/off codes (the machine's own slots, the input, scratch, the
    output). Position: for the input, the patch's row and column written smoothly (sines and
    cosines at three scales, 6 numbers each), so a neighbouring patch has a nearby address
    and "one to the right" is the same small step everywhere. Elsewhere, a random vector."""
    side = math.isqrt(n_in)
    slots = head + n_in + scratch + 1
    where = torch.zeros(slots, SLOT)
    region = torch.eye(4) * 2.0
    where[:head, :4], where[head:head + n_in, :4] = region[0], region[1]
    where[head + n_in:-1, :4], where[-1, :4] = region[2], region[3]
    where[:, 4:] = torch.randn(slots, 12, generator=g) * 0.5
    rc = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"),
                     dim=-1).view(-1, 2).float()
    feats = []
    for axis in range(2):
        for scale in (1.0, 2.0, 4.0):
            a = rc[:, axis] * math.pi / (side * scale)
            feats += [torch.sin(a), torch.cos(a)]
    where[head:head + n_in, 4:] = torch.stack(feats, dim=1)
    return where


class Core(nn.Module):
    def __init__(self, n_in=49, scratch=16, ops=256, k=4, ticks=16, addresses="random",
                 reads=1, see_pc=False, top=0, jitter=0.0, see_head=False):
        super().__init__()
        # see_head: the vop also sees the pc and the address slots as they stand, so it can
        # step a pointer ("one to the right of where I am"); without it a vop cannot know
        # where its pointers are
        self.see_head = see_head
        self.jitter = jitter  # training only: noise added to every address before it points
        self.n_in, self.scratch, self.ops, self.k, self.ticks = n_in, scratch, ops, k, ticks
        self.reads, self.see_pc, self.top = reads, see_pc, top
        self.head = head = reads + 2                      # pc, the read addresses, write addr
        slots = head + n_in + scratch + 1
        g = torch.Generator().manual_seed(1)
        where = torch.randn(slots, SLOT, generator=g)
        if addresses == "grid":
            where = grid_addresses(head, n_in, scratch, g)
        self.register_buffer("where", F.normalize(where, dim=1))
        can = torch.ones(slots)
        can[head:head + n_in] = 0                         # the world's input cannot be written
        self.register_buffer("can_write", can)
        self.boot = nn.Parameter(F.normalize(torch.randn(head, SLOT), dim=1) * 0.5)
        self.keys = nn.Parameter(torch.randn(ops, SLOT) / math.sqrt(SLOT))
        fan_in = (reads + 1 + int(see_pc) + (head if see_head else 0)) * SLOT
        fan_out = (1 + head) * SLOT                       # v, then the next head slots
        self.m = nn.Parameter(torch.randn(ops, fan_out, fan_in) / math.sqrt(fan_in))
        self.bias = nn.Parameter(torch.randn(ops, fan_out) * 0.1)
        self.sharp = nn.Parameter(torch.tensor(8.0))      # how sharply an address picks slots
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.register_buffer("picks", torch.zeros(ops))
        self.trace, self.answers = None, None

    def point(self, addr):
        """Address vectors [B, n, 16] -> how much each slot answers, [B, n, slots]."""
        addr = F.normalize(addr, dim=-1)
        if self.training and self.jitter > 0:
            addr = F.normalize(addr + self.jitter * torch.randn_like(addr), dim=-1)
        match = self.sharp * (addr @ self.where.T)
        if self.top:
            floor = match.topk(self.top, dim=-1).values[..., -1:]
            match = match.masked_fill(match < floor, float("-inf"))
        return torch.softmax(match, dim=-1)

    def forward(self, x, trace=False):
        n, head = len(x), self.head
        mem = torch.cat([self.boot.expand(n, -1, -1), patches(x),
                         x.new_zeros(n, self.scratch + 1, SLOT)], dim=1)
        log, answers = [], []
        for _ in range(self.ticks):
            pc = F.layer_norm(mem[:, 0], (SLOT,))
            sc, idx = (pc @ self.keys.T).topk(self.k, dim=-1)                 # fetch
            mix = torch.softmax(sc, dim=-1)
            aim = self.point(mem[:, 1:head])                                 # [B, R + 1, slots]
            got = torch.einsum("bns,bsd->bnd", aim, mem).flatten(1)          # load
            if self.see_pc:
                got = torch.cat([got, mem[:, 0]], dim=1)
            if self.see_head:
                mine = mem[:, :head]
                if self.see_head == "other":   # a control: another run's pointers, same size
                    mine = torch.roll(mine, 1, dims=0)
                got = torch.cat([got, mine.flatten(1)], dim=1)
            out = F.hardtanh(torch.einsum("bkij,bj->bki", self.m[idx], got) + self.bias[idx])
            out = torch.einsum("bk,bki->bi", mix, out).view(n, 1 + head, SLOT)   # execute
            w = aim[:, -1]
            share = (w * self.can_write).unsqueeze(-1)                       # store
            mem = move(mem, out[:, :1], share)
            mem = torch.cat([move(mem[:, :head], out[:, 1:]), mem[:, head:]], dim=1)
            answers.append(mem[:, -1, :10] * self.scale)
            if not self.training:
                self.picks += torch.bincount(idx[:, 0], minlength=self.ops).float()
            if trace:
                log.append((idx[:, 0], aim[:, :-1].detach(), w.detach()))
        self.trace, self.answers = log, answers
        return answers[-1]

    def madds_total(self):
        slots = self.where.shape[0]
        fan_in = (self.reads + 1 + int(self.see_pc) + (self.head if self.see_head else 0)) * SLOT
        return self.ticks * (self.ops * SLOT + (self.reads + 1) * slots * SLOT * 2 +
                             self.k * (1 + self.head) * SLOT * fan_in)


def every_tick(model, xb, yb):
    """The answer is asked for after every tick: the mean of the usual loss over ticks."""
    model(xb)
    return sum(F.cross_entropy(a, yb) for a in model.answers) / len(model.answers)


def report(label, model, data, fit, test):
    """One row: scores, vops run, and how much of the image the reads cover: the number of
    input patches that got at least 5% of one read, summed over the run, and how many slots
    one read is spread over (1 is a point, 49 is the whole image)."""
    model.picks.zero_()
    score(model, data[2], data[3], "class")
    model.eval()
    with torch.no_grad():
        model(data[2][:1000], trace=True)
    reads = torch.stack([t[1] for t in model.trace], dim=1)       # [n, ticks, R, slots]
    on_input = reads[..., model.head:model.head + model.n_in]
    covered = (on_input.flatten(1, 2).max(dim=1).values > 0.05).float().sum(dim=1).mean()
    spread = torch.exp(-(reads * torch.log(reads.clamp_min(1e-9))).sum(dim=-1)).mean()
    print("| %s | %d | %d | %d | %.3f | %.3f | %d of %d | %.0f of %d | %.1f | %.1f |" % (
        label, model.ticks, params(model), model.madds_total(), fit, test,
        int((model.picks > 0).sum()), model.ops, covered, model.n_in, spread,
        model.sharp.item()), flush=True)


HEAD = ("| what | ticks | params | madds | train | test | vops run | patches read a run | "
        "slots one read is spread over | sharpness |\n|---|---|---|---|---|---|---|---|---|---|")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    torch.set_num_threads(8)
    if arg not in ("checks", "explore", "anneal", "best"):
        data = TASKS[arg](0)
        print("%s, the vector virtual machine v6, 256 vops, 4 blended a tick; 6,000 steps" % arg)
        print(HEAD, flush=True)
        for ticks in (8, 16, 32):
            torch.manual_seed(0)
            model = Core(ticks=ticks)
            fit, test, _ = train(model, data, 6000, 128, 3e-3, 0, wd=0.05, shift=2)
            report("as designed", model, data, fit, test)
        return
    if arg == "explore":
        data = TASKS["mnist"](0)
        print("mnist, version 6 at 8 ticks: is it an exploration problem? more steps, and "
              "noise on the addresses")
        print(HEAD, flush=True)
        for label, kw, steps in [("6,000 steps", {}, 6000),
                                 ("6,000 steps, address noise 0.3", {"jitter": 0.3}, 6000),
                                 ("6,000 steps, address noise 0.6", {"jitter": 0.6}, 6000),
                                 ("24,000 steps", {}, 24000),
                                 ("24,000 steps, address noise 0.3", {"jitter": 0.3}, 24000)]:
            torch.manual_seed(0)
            model = Core(**{"ticks": 8, **kw})
            fit, test, _ = train(model, data, steps, 128, 3e-3, 0, wd=0.05, shift=2)
            report(label, model, data, fit, test)
        return
    if arg == "best":
        data = TASKS["mnist"](0)
        print("mnist, version 6 at 8 ticks: the changes that helped, together; 12,000 steps")
        print(HEAD, flush=True)
        for label, kw in [("4 reads, sees its pointers", {"reads": 4, "see_head": True}),
                          ("8 reads, sees its pointers", {"reads": 8, "see_head": True}),
                          ("4 reads, sees its pointers, 16 ticks",
                           {"reads": 4, "see_head": True, "ticks": 16})]:
            torch.manual_seed(0)
            model = Core(**{"ticks": 8, **kw})
            fit, test, _ = train(model, data, 12000, 128, 3e-3, 0, wd=0.05, shift=2)
            report(label, model, data, fit, test)
        return
    if arg == "anneal":
        data = TASKS["mnist"](0)
        print("mnist, version 6 at 8 ticks: address noise on a curriculum (it starts at the "
              "strength given and falls in a straight line to nothing at 70% of training)")
        print(HEAD, flush=True)
        for start, steps in ((0.3, 6000), (0.6, 6000), (1.0, 6000), (0.6, 24000)):
            def fade(model, done, start=start):
                model.jitter = start * max(0.0, 1.0 - done / 0.7)
            torch.manual_seed(0)
            model = Core(ticks=8, jitter=start)
            fit, test, _ = train(model, data, steps, 128, 3e-3, 0, wd=0.05, shift=2,
                                 each_step=fade)
            report("noise from %.1f, %d steps" % (start, steps), model, data, fit, test)
        return
    data = TASKS["mnist"](0)
    print("mnist, version 6 at 8 ticks, one change at a time; 6,000 steps")
    print(HEAD, flush=True)
    for label, kw, loss in [
        ("as designed", {}, None),
        ("4 reads a tick", {"reads": 4}, None),
        ("the vop sees the pc", {"see_pc": True}, None),
        ("answer asked every tick", {}, every_tick),
        ("4 reads + every tick", {"reads": 4}, every_tick),
        ("4 reads + every tick, 16 ticks", {"reads": 4, "ticks": 16}, every_tick),
    ]:
        torch.manual_seed(0)
        model = Core(**{"ticks": 8, **kw})
        fit, test, _ = train(model, data, 6000, 128, 3e-3, 0, wd=0.05, shift=2, loss_fn=loss)
        report(label, model, data, fit, test)


if __name__ == "__main__":
    main()
