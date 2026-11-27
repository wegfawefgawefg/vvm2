"""The lights game (Simon) around the core.

    python -m old.simon8 [STEPS] [hold=1 lag=1 max=6 colours=4 scratch=64 ops=32 seed=0
                                  gru=1 threads=1 mix=0 loop=0
                                  before=0 grow=0 draw=0]

The world owns two input slots and reads one output slot:

    light    the colour lit now (silent once the lights are over)
    signal   "watch" while the lights are shown, then "play"
    button   the output slot; the world reads it once for each light, in order

Timing, in ticks. Every light is lit for `hold` ticks, with no gap between lights. The signal
turns to "play" on the tick after the last light. The world reads the button for light i
after tick  hold * length + lag + hold * i:  `lag` ticks after the signal turns, then one
read every `hold` ticks. With lag 0 the first answer is wanted on the very tick the machine
first sees "play", before a pointer can have moved, so a machine that works by moving
pointers needs a lag of at least 1.

loop=1 starts every vop with a silent next-pc, so the pc stays where it is and the same
vops run again until one learns to jump: looping is the default and jumping is learned.
before=2000 has the same machine learn to copy lists for that many steps first, then the
lights game; copy is scored before and after, to see what is kept.
draw=1 keeps the pointers exact while training as well (see old/core8.py).
grow=1 is a curriculum of data: games of 1 to 2 lights at first, the longest game growing
to `max` by 60% of the way through training.
mix=1 trains on two runs of every batch, one as usual and one with whole steps, and adds
the two losses.

Trained on lengths 1 to `max`, tested on longer ones, as trained and run with whole steps
(exact pointers). A GRU of the same weight count sees the same two slots tick by tick.
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from vvm2.common import params
from .core8 import SLOT, Core
from .tape8 import MOVES, codebook, sized

def frames(book, x, hold, lag):
    """x [B, L] colours -> the two input slots at every tick [B, ticks, 2, 16], and the
    ticks after which the button is read. The last two rows of the book are watch, play."""
    n, length = x.shape
    reads = [hold * length + lag + hold * i for i in range(length)]
    ticks = reads[-1] + 1
    light = torch.zeros(n, ticks, SLOT)
    light[:, :hold * length] = book[x].repeat_interleave(hold, dim=1)
    signal = book[-1].expand(n, ticks, -1).clone()
    signal[:, :hold * length] = book[-2]
    return torch.stack([light, signal], dim=2), reads


class SimonVM(nn.Module):
    """The machine and the world's code book. Nothing is learned but the machine and one
    number, how sharply the button is read."""

    def __init__(self, colours=4, hold=1, lag=1, **core):
        super().__init__()
        self.colours, self.hold, self.lag = colours, hold, lag
        self.core = Core(**core)
        self.register_buffer("book", codebook(colours + 2))
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, trace=False):
        f, reads = frames(self.book, x, self.hold, self.lag)
        outs = self.core(lambda t: f[:, t], f.shape[1], 1, trace=trace)
        return outs[reads, :, 0].transpose(0, 1) @ self.book[:self.colours].T * self.temp


class SimonGRU(nn.Module):
    def __init__(self, width, colours=4, hold=1, lag=1):
        super().__init__()
        self.hold, self.lag = hold, lag
        self.register_buffer("book", codebook(colours + 2))
        self.gru = nn.GRU(2 * SLOT, width, batch_first=True)
        self.out = nn.Linear(width, colours)

    def forward(self, x):
        f, reads = frames(self.book, x, self.hold, self.lag)
        y, _ = self.gru(f.flatten(2))
        return self.out(y[:, reads])


def copy_logits(vm, x):
    """Another task on the same machine: copy a list from the input slots to the output
    slots (as old/tape8.py does), with the colours as the symbols."""
    outs = vm.core(vm.book[x], 2 * x.shape[1] + 4, x.shape[1])
    return outs[-1] @ vm.book[:vm.colours].T * vm.temp


def copy_score(vm, colours, when):
    g, cells = torch.Generator().manual_seed(11), []
    vm.eval()
    vm.core.hard = True
    with torch.no_grad():
        for length in (8, 24, 48):
            x = torch.randint(colours, (200, length), generator=g)
            hit = copy_logits(vm, x).argmax(dim=-1) == x
            cells.append("%.2f / %.2f" % (hit.float().mean(), hit.all(dim=1).float().mean()))
    vm.core.hard = False
    print("    copy with whole steps %s, lengths 8, 24, 48: %s" % (when, " | ".join(cells)),
          flush=True)


def show_trace(vm, x, hard):
    """Run 0 of a batch, tick by tick: the vop that ran (the best match), where each pointer
    is after the tick and the move that took it there. i0 is the light, i1 the signal, s3
    the fourth scratch slot, o0 the button."""
    core = vm.core
    core.hard = hard
    with torch.no_grad():
        said = vm(x, trace=True).argmax(dim=-1)
    core.hard = False
    log, n_in = core.trace, 2
    turn = vm.hold * x.shape[1]

    def name(i):
        if i == 0:
            return "pc"
        if i <= n_in:
            return "i%d" % (i - 1)
        if i <= n_in + core.scratch:
            return "s%d" % (i - 1 - n_in)
        return "o%d" % (i - 1 - n_in - core.scratch)

    print("  trace, %s: lights %s, answered %s; pointers exact to %.2f" % (
        "run with whole steps" if hard else "as trained",
        " ".join(str(int(c)) for c in x[0]), " ".join(str(int(c)) for c in said[0]),
        core.sharp()))
    print("    signal:  " + " ".join("%5s" % ("watch" if t < turn else "PLAY") for t in range(len(log))))
    print("    vop:     " + " ".join("%5d" % int(t[0][0]) for t in log))
    for p in range(core.reads + 1):
        label = "write:   " if p == core.reads else "read %d:  " % (p + 1)
        print("    " + label + " ".join("%5s" % name(int(t[1][0, p].argmax())) for t in log))
        print("      did:   " + " ".join("%5s" % MOVES[int(t[2][0, p].argmax())] for t in log))
    sys.stdout.flush()


def score(model, tests, g, colours):
    cells = []
    with torch.no_grad():
        for length in tests:
            x = torch.randint(colours, (300, length), generator=g)
            hit = model(x).argmax(dim=-1) == x
            cells.append("%.2f / %.2f" % (hit.float().mean(), hit.all(dim=1).float().mean()))
    return " | ".join(cells)


def run(steps=8000, hold=1, lag=1, most=6, colours=4, scratch=64, ops=32, seed=0, gru=True,
        threads=1, mix=False, loop=False, before=0, grow=False, draw=False):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    vm = SimonVM(colours, hold, lag, scratch=scratch, ops=ops, k=min(4, ops))
    vm.core.draw = draw
    if loop:   # every vop starts out leaving the pc alone, so the same vops come round
        with torch.no_grad():   # again until one learns to jump: a loop is the default
            vm.core.m[:, SLOT:2 * SLOT] = 0
            vm.core.bias[:, SLOT:2 * SLOT] = 0
    # a routine that stores a light a tick needs hold * length scratch slots and one spare
    tests = [t for t in (most, 2 * most, 4 * most, 8 * most) if hold * t < scratch]
    make = lambda w: SimonGRU(w, colours, hold, lag)   # noqa: E731
    print("simon: %d colours, %d tick%s a light, first read %d tick%s after the signal turns; "
          "trained on lengths 1 to %d, %d steps, seed %d. In each cell: share of lights right "
          "/ share of whole games right%s" % (
              colours, hold, "" if hold == 1 else "s", lag, "" if lag == 1 else "s", most,
              steps, seed, (". Trained on the run with whole steps too" if mix else "")
              + (". Vops start out leaving the pc alone" if loop else "")
              + (". The longest game grows from 2" if grow else "")
              + (". Pointers exact while training" if draw else "")))
    print("| model | weights | " + " | ".join("length %d" % t for t in tests) + " |")
    print("|---|---|" + "---|" * len(tests), flush=True)
    if before:   # the same machine learns to copy lists first (lengths 2 to 8)
        opt = torch.optim.AdamW(vm.parameters(), lr=2e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=before)
        g = torch.Generator().manual_seed(seed + 1)
        for _ in range(before):
            vm.train()
            x = torch.randint(colours, (64, int(torch.randint(2, 9, (1,), generator=g))), generator=g)
            loss = F.cross_entropy(copy_logits(vm, x).reshape(-1, colours), x.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vm.parameters(), 1.0)
            opt.step()
            sched.step()
        copy_score(vm, colours, "after %d steps of copy" % before)
    models = [("vvm", vm)] + ([("gru", make(sized(make, params(vm))))] if gru else [])
    for name, model in models:
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        g = torch.Generator().manual_seed(seed)
        for step in range(steps):
            if name == "vvm" and step and step % max(1, steps // 8) == 0:   # an early look
                model.eval()
                vm.core.hard = True
                whole = score(vm, tests[:2], torch.Generator().manual_seed(5), colours)
                vm.core.hard = False
                print("    step %d: loss %.3f; with whole steps, lengths %s: %s" % (
                    step, loss.item(), ", ".join(str(t) for t in tests[:2]), whole), flush=True)
            model.train()
            top = most if not grow else min(most, 2 + int((most - 2) * step / (0.6 * steps)))
            length = int(torch.randint(1, top + 1, (1,), generator=g))
            x = torch.randint(colours, (64, length), generator=g)
            loss = F.cross_entropy(model(x).reshape(-1, colours), x.reshape(-1))
            if mix and name == "vvm":   # the same batch run with whole steps as well: the
                model.eval()            # gradient there reaches what is stored and the pc
                vm.core.hard = True
                loss = loss + F.cross_entropy(model(x).reshape(-1, colours), x.reshape(-1))
                vm.core.hard = False
                model.train()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        print("| %s | %d | %s |" % (name, params(model), score(model, tests, g, colours)), flush=True)
        if name == "vvm":
            vm.core.hard = True
            print("| vvm, run with whole steps | %d | %s |" % (
                params(vm), score(vm, tests, g, colours)), flush=True)
            vm.core.hard = False
    if before:
        copy_score(vm, colours, "after the lights game")
    x = torch.randint(colours, (8, min(4, most)), generator=torch.Generator().manual_seed(1))
    show_trace(vm, x, hard=False)
    show_trace(vm, x, hard=True)


if __name__ == "__main__":
    a = dict(s.split("=") for s in sys.argv[1:] if "=" in s)
    first = [s for s in sys.argv[1:] if "=" not in s]
    run(steps=int(first[0]) if first else 8000, hold=int(a.get("hold", 1)),
        lag=int(a.get("lag", 1)), most=int(a.get("max", 6)), colours=int(a.get("colours", 4)),
        scratch=int(a.get("scratch", 64)), ops=int(a.get("ops", 32)), seed=int(a.get("seed", 0)),
        gru=a.get("gru", "1") == "1", threads=int(a.get("threads", 1)), mix=a.get("mix", "0") == "1", loop=a.get("loop", "0") == "1",
        before=int(a.get("before", 0)), grow=a.get("grow", "0") == "1",
        draw=a.get("draw", "0") == "1")
