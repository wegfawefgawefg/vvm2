"""Lists: copy and reverse, tested on longer lists than trained.

    python -m old.tape8 copy|reverse [STEPS] [ops=32 seed=0 gru=1 draw=0]

The world puts the list one item a slot in the input region and reads the answer from as
many output slots. An item is one of 8 symbols, each a fixed vector of 16 numbers (the
world's code book; the machine never learns it). The answer in an output slot is whichever
symbol's vector it is nearest. The machine gets 2 ticks an item plus 4. Trained on lengths 2
to 8, tested on 8 to 96, as trained and run with whole steps (exact pointers). A GRU of the
same weight count reads the list and is asked for the answers one by one.

draw=1 keeps the pointers exact while training too (see old/core8.py).
(Lookup by key and the other tasks of the form before this one are in old/tape7.py.)
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from vvm2.common import params
from .core8 import SLOT, Core

SYMBOLS = 8
MOVES = ["stay", "next", "prev"]


def codebook(n, seed=7):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, SLOT, generator=g), dim=1) * 0.9


class TapeVM(nn.Module):
    """The machine with the world's code book around it. Nothing here is learned except
    the machine itself and one number, how sharply an answer is read."""

    def __init__(self, **core):
        super().__init__()
        self.core = Core(**core)
        self.register_buffer("book", codebook(SYMBOLS))
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, tokens, trace=False):
        """tokens [B, L] -> logits [B, L, SYMBOLS] read from the output slots at the end."""
        length = tokens.shape[1]
        outs = self.core(self.book[tokens], 2 * length + 4, length, trace=trace)
        return outs[-1] @ self.book.T * self.temp


class ListGRU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer("book", codebook(SYMBOLS))
        self.read = nn.GRU(SLOT, width, batch_first=True)
        self.write = nn.GRU(1, width, batch_first=True)
        self.out = nn.Linear(width, SYMBOLS)

    def forward(self, tokens):
        _, h = self.read(self.book[tokens])
        y, _ = self.write(torch.zeros(tokens.shape[0], tokens.shape[1], 1), h)
        return self.out(y)


def sized(make, target):
    """The width at which a model has about `target` weights."""
    return min(range(8, 600, 4), key=lambda w: abs(params(make(w)) - target))


def place(i, n_in, scratch):
    """A slot number as a region letter and a number: pc, i3, s0, o2."""
    if i == 0:
        return "pc"
    if i <= n_in:
        return "i%d" % (i - 1)
    if i <= n_in + scratch:
        return "s%d" % (i - 1 - n_in)
    return "o%d" % (i - 1 - n_in - scratch)


def show_trace(vm, x):
    """Run 0 with whole steps, tick by tick: the vop that ran (the best match), where each
    pointer is after the tick and the move that took it there."""
    core = vm.core
    core.hard = True
    with torch.no_grad():
        vm(x, trace=True)
    core.hard = False
    log, n_in = core.trace, x.shape[1]
    print("  trace, length %d, whole steps" % n_in)
    print("    vop:     " + " ".join("%4d" % int(t[0][0]) for t in log))
    for p in range(core.reads + 1):
        label = "write:   " if p == core.reads else "read %d:  " % (p + 1)
        print("    " + label + " ".join("%4s" % place(int(t[1][0, p].argmax()), n_in, core.scratch)
                                       for t in log))
        print("      did:   " + " ".join("%4s" % MOVES[int(t[2][0, p].argmax())] for t in log))
    sys.stdout.flush()


def score(model, tests, g, reverse):
    cells = []
    with torch.no_grad():
        for length in tests:
            x = torch.randint(SYMBOLS, (300, length), generator=g)
            hit = model(x).argmax(dim=-1) == (x.flip(1) if reverse else x)
            cells.append("%.2f / %.2f" % (hit.float().mean(), hit.all(dim=1).float().mean()))
    return " | ".join(cells)


def run(task, steps=3000, ops=32, seed=0, gru=True, draw=False):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    reverse, tests = task == "reverse", (8, 12, 16, 24, 48, 96)
    vm = TapeVM(ops=ops, k=min(4, ops))
    vm.core.draw = draw
    print("%s: trained on lengths 2 to 8, %d steps, a bank of %d vops, seed %d%s. In each cell: "
          "share of items right / share of whole lists right" % (
              task, steps, ops, seed, ", pointers exact while training" if draw else ""))
    print("| model | weights | " + " | ".join("length %d" % t for t in tests) + " |")
    print("|---|---|" + "---|" * len(tests), flush=True)
    models = [("vvm", vm)] + ([("gru", ListGRU(sized(ListGRU, params(vm))))] if gru else [])
    for name, model in models:
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        g = torch.Generator().manual_seed(seed)
        for _ in range(steps):
            model.train()
            x = torch.randint(SYMBOLS, (64, int(torch.randint(2, 9, (1,), generator=g))), generator=g)
            want = x.flip(1) if reverse else x
            loss = F.cross_entropy(model(x).reshape(-1, SYMBOLS), want.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        print("| %s | %d | %s |" % (name, params(model), score(model, tests, g, reverse)), flush=True)
        if name == "vvm":
            vm.core.hard = True
            print("| vvm, run with whole steps | %d | %s |" % (
                params(vm), score(vm, tests, g, reverse)), flush=True)
            vm.core.hard = False
            show_trace(vm, torch.randint(SYMBOLS, (4, 8), generator=g))


if __name__ == "__main__":
    a = dict(s.split("=") for s in sys.argv[2:] if "=" in s)
    first = [s for s in sys.argv[2:] if "=" not in s]
    run(sys.argv[1], steps=int(first[0]) if first else 3000, ops=int(a.get("ops", 32)),
        seed=int(a.get("seed", 0)), gru=a.get("gru", "1") == "1", draw=a.get("draw", "0") == "1")
