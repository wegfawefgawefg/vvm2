"""Lists: copy and reverse, tested on longer lists than trained.

    python -m vvm2.tape copy|reverse [STEPS] [vops=32 cores=4 seed=0 gru=1 mix=1]

The world puts the list one item a slot in the input slots and reads the answer from as
many output slots. An item is one of 8 symbols, each a fixed vector of 16 numbers (the
world's code book; the machine never learns it). The answer in an output slot is whichever
symbol's vector it is nearest. The machine gets 2 ticks an item plus 4. Trained on lengths 2
to 8, tested on 8 to 96, as trained and run with whole steps (exact pointers). A GRU of
about the same weight count reads the list and is asked for the answers one by one.

mix=1 (the default) trains on two runs of every batch, one as usual and one with whole
steps, and adds the two losses.
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import params
from .core import SLOT, Core

SYMBOLS = 8


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
    return min(range(4, 600, 2), key=lambda w: abs(params(make(w)) - target))


def show_trace(vm, x, n_out, ticks=None):
    """Run 0 with whole steps, tick by tick, for every core: what its pc matched best, where
    its read and write pointers are after the tick, and the steps its control slot gave."""
    core = vm.core
    core.hard = True
    with torch.no_grad():
        vm(x, trace=True)
    core.hard = False
    log = core.trace[:ticks]
    n_in = core.layout(0, 0)[4]          # the size with no input or output...
    n_in = int(log[0][1].shape[-1]) - n_in - n_out   # ...gives the number of input slots
    name = lambda i: core.name(int(i), n_in, n_out)  # noqa: E731
    print("  trace, whole steps, run 0 (a step: +1 next, -1 previous, 0 stay)")
    for c in range(core.cores):
        print("    core %d  runs:  " % c + " ".join("%5s" % name(t[0][0, c]) for t in log))
        for p, label in ((0, "read: "), (1, "write:")):
            print("            %s " % label + " ".join("%5s" % name(t[1][0, c, p].argmax()) for t in log))
            print("              step: " + " ".join("%+5d" % int(t[2][0, c, p]) for t in log))
    sys.stdout.flush()


def score(model, tests, g, reverse):
    cells = []
    with torch.no_grad():
        for length in tests:
            x = torch.randint(SYMBOLS, (300, length), generator=g)
            hit = model(x).argmax(dim=-1) == (x.flip(1) if reverse else x)
            cells.append("%.2f / %.2f" % (hit.float().mean(), hit.all(dim=1).float().mean()))
    return " | ".join(cells)


def run(task, steps=3000, vops=32, cores=4, seed=0, gru=True, mix=True):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    reverse, tests = task == "reverse", (8, 12, 16, 24, 48, 96)
    vm = TapeVM(vops=vops, cores=cores)
    print("%s: trained on lengths 2 to 8, %d steps, %d cores, %d vops, seed %d%s. In each cell: "
          "share of items right / share of whole lists right" % (
              task, steps, cores, vops, seed, ", the whole-step run in the loss too" if mix else ""))
    print("| model | weights | " + " | ".join("length %d" % t for t in tests) + " |")
    print("|---|---|" + "---|" * len(tests), flush=True)
    models = [("vvm", vm)] + ([("gru", ListGRU(sized(ListGRU, params(vm))))] if gru else [])
    for name, model in models:
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        g = torch.Generator().manual_seed(seed)
        for step in range(steps):
            model.train()
            x = torch.randint(SYMBOLS, (64, int(torch.randint(2, 9, (1,), generator=g))), generator=g)
            want = x.flip(1) if reverse else x
            loss = F.cross_entropy(model(x).reshape(-1, SYMBOLS), want.reshape(-1))
            if mix and name == "vvm":
                model.eval()
                vm.core.hard = True
                loss = loss + F.cross_entropy(model(x).reshape(-1, SYMBOLS), want.reshape(-1))
                vm.core.hard = False
                model.train()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if name == "vvm" and step % max(1, steps // 6) == 0:
                print("    step %d: loss %.3f" % (step, loss.item()), flush=True)
        model.eval()
        print("| %s | %d | %s |" % (name, params(model), score(model, tests, g, reverse)), flush=True)
        if name == "vvm":
            vm.core.hard = True
            print("| vvm, run with whole steps | %d | %s |" % (
                params(vm), score(vm, tests, g, reverse)), flush=True)
            vm.core.hard = False
            show_trace(vm, torch.randint(SYMBOLS, (4, 6), generator=g), 6)


if __name__ == "__main__":
    a = dict(s.split("=") for s in sys.argv[2:] if "=" in s)
    first = [s for s in sys.argv[2:] if "=" not in s]
    run(sys.argv[1], steps=int(first[0]) if first else 3000, vops=int(a.get("vops", 32)),
        cores=int(a.get("cores", 4)), seed=int(a.get("seed", 0)), gru=a.get("gru", "1") == "1",
        mix=a.get("mix", "1") == "1")
