"""Ordinary curves on the frozen core: xor of two numbers, a sine, two spirals.

  python -m vvm2.curves TASK [STEPS] [key=value ...]      TASK: xor | sin | spirals | all
  options:  layout=one   one input slot holding the numbers in its first places (the default)
            layout=each  one input slot a number, the number in the slot's first place
            ticks=4      ticks the machine runs before the answer is read
            ops=16  k=4  the size of the bank and how many vops are fetched
            reads=2      read pointers (reads=1 with layout=each: it must step to see both)
            gain=1       sine only: the answer is `gain` times the output slot's first number
            seeds=3      how many seeds (0, 1, ...)
            trace=0      skip the tick-by-tick trace
  python -m vvm2.curves --quick                           every task, 200 steps, one seed

How the numbers get in and out. There is no learned encoder and no readout. The driver
lays the numbers out raw: a slot is 16 numbers, the input's numbers go in the first places
and the rest are 0. The answer is read from the one output slot after the last tick: for
two classes, whichever of its first two numbers is larger (in training, those two numbers
times one learned number, how sharply the answer is read, as in tape.py); for the sine, its
first number. A slot can never be longer than 1 (the write rule), so sin = 1 needs the slot
to be exactly (1, 0, 0, ...); `gain=2` reads twice the first number to take that strain off.

Beside the machine: an MLP (two hidden layers, vvm2/common.py) with the same number of
weights, and one with the same multiply-adds a run. The machine's multiply-adds are counted
as ticks x (bank size x 16 for the fetch + k x the vop matrix for the execute).
The data sets and the training loop are the ones in vvm2/common.py, the same for both.
"""
import math
import sys

import torch
import torch.nn as nn

from .common import TASKS, mlp_matched, params, score, train
from .core import SLOT, Core


def lay(x, layout):
    """[B, d] numbers -> input slots. 'one': [B, 1, 16], the numbers in the first d places.
    'each': [B, d, 16], number i in the first place of slot i."""
    n, d = x.shape
    if layout == "one":
        slots = x.new_zeros(n, 1, SLOT)
        slots[:, 0, :d] = x
    else:
        slots = x.new_zeros(n, d, SLOT)
        slots[:, :, 0] = x
    return slots


class CurveVM(nn.Module):
    """The machine with a driver around it. Learned: the machine, and for classes one
    number (how sharply the answer is read)."""

    def __init__(self, kind, layout="one", ticks=4, gain=1.0, **core):
        super().__init__()
        self.core = Core(**core)
        self.kind, self.layout, self.ticks, self.gain = kind, layout, ticks, gain
        if kind == "class":
            self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, trace=False):
        out = self.core(lay(x, self.layout), self.ticks, 1, trace=trace)[-1][:, 0]
        return out[:, :2] * self.temp if self.kind == "class" else out[:, :1] * self.gain

    def madds(self):
        c = self.core
        fan_in = (c.reads + 2) * SLOT
        return self.ticks * (c.ops * SLOT + c.k * c.fan_out * fan_in)


def place(i, n_in, scratch):
    """A slot number as a region letter and a number: pc, i0, s3, o0."""
    if i == 0:
        return "pc"
    if i <= n_in:
        return "i%d" % (i - 1)
    if i <= n_in + scratch:
        return "s%d" % (i - 1 - n_in)
    return "o%d" % (i - 1 - n_in - scratch)


def bank_use(log, ops):
    """From a trace (one entry a tick, the best-matching vop of every run): the share of the
    bank ever picked, the entropy of use (1.0: all vops equally), the busiest vop's share,
    and how the picks change with the input: how many different paths (the sequence of
    picks over the ticks) the runs took, and the share of runs on the commonest path."""
    picks = torch.stack([t[0] for t in log], dim=1)                    # [B, ticks]
    count = torch.bincount(picks.flatten(), minlength=ops).float()
    p = count / count.sum()
    entropy = -(p[p > 0] * p[p > 0].log()).sum().item() / math.log(max(ops, 2))
    paths = {}
    for row in picks.tolist():
        paths[tuple(row)] = paths.get(tuple(row), 0) + 1
    return ("bank: %.2f of %d vops used, entropy %.2f, busiest %.2f; %d paths, the commonest "
            "taken by %.2f of runs" % ((count > 0).float().mean(), ops, entropy, p.max(),
                                       len(paths), max(paths.values()) / len(picks)))


MOVES = ["stay", "next", "prev", "in0", "inN", "sc0", "scN", "out0", "outN", "seek", "twin"]


def show_trace(vm, x, labels):
    """A few runs tick by tick, with whole steps: the vop that ran, where each pointer is
    after the tick and the move that took it there."""
    vm.core.hard = True
    with torch.no_grad():
        vm(x, trace=True)
    vm.core.hard = False
    core, log = vm.core, vm.core.trace
    n_in = 1 if vm.layout == "one" else x.shape[1]
    for b, label in enumerate(labels):
        print("    %s" % label)
        print("      vop:    " + " ".join("%5d" % int(t[0][b]) for t in log))
        for p in range(core.reads + 1):
            name = "write:  " if p == core.reads else "read %d: " % (p + 1)
            print("      " + name + " ".join(
                "%5s" % place(int(t[1][b, p].argmax()), n_in, core.scratch) for t in log))
            print("        did:  " + " ".join("%5s" % MOVES[int(t[2][b, p].argmax())] for t in log))
    sys.stdout.flush()


def probes(task, tx):
    """A few inputs worth tracing."""
    if task == "xor":
        x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    elif task == "sin":
        x = torch.tensor([[-math.pi / 2], [0.0], [math.pi / 2], [3.0]])
    else:
        x = tx[:3]
    return x, ["input " + ", ".join("%.2f" % v for v in row) for row in x.tolist()]


def fmt(kind, v):
    return "%.3f" % v if kind == "class" else "%.5f" % v


def run(task, steps, layout="one", ticks=4, ops=16, k=4, reads=2, gain=1.0, seeds=3,
        batch=64, lr=2e-3, trace=True):
    torch.set_num_threads(4)
    rows = {}
    kind = None
    for seed in range(seeds):
        data = TASKS[task](seed)
        x, y, tx, ty, kind = data
        d, outs = x.shape[1], 2 if kind == "class" else 1
        torch.manual_seed(seed)
        vm = CurveVM(kind, layout, ticks, gain, ops=ops, k=min(k, ops), reads=reads, scratch=4)
        tr, te, _ = train(vm, data, steps, batch, lr, seed)
        rows.setdefault("vvm, as trained", (params(vm), vm.madds(), []))[2].append((tr, te))
        vm.core.hard = True
        hard = (score(vm, x, y, kind), score(vm, tx, ty, kind))
        rows.setdefault("vvm, run with whole steps", (params(vm), vm.madds(), []))[2].append(hard)
        with torch.no_grad():
            vm(tx, trace=True)
            use = bank_use(vm.core.trace, ops)
            vm.core.hard = False
            vm(tx)
        use += "; pointers exact to %.2f as trained" % vm.core.sharp()
        for name, target in (("mlp, same weights", params(vm)), ("mlp, same multiply-adds", None)):
            torch.manual_seed(seed)
            if target is None:   # an MLP's multiply-adds are its weights
                target = vm.madds()
            mlp = mlp_matched(d, outs, target)
            tr, te, _ = train(mlp, data, steps, batch, lr, seed)
            rows.setdefault(name, (params(mlp), mlp.madds_total(), []))[2].append((tr, te))
        print("  seed %d: %s" % (seed, use), flush=True)
        if trace and seed == 0:
            show_trace(vm, *probes(task, tx))
    what = "share right" if kind == "class" else "mean squared error"
    print("%s, layout '%s', %d ticks, bank of %d (k %d), %d read pointers, %d steps, %d seed%s. "
          "%s, mean of seeds (each seed)" % (task, layout, ticks, ops, min(k, ops), reads, steps,
                                             seeds, "" if seeds == 1 else "s", what))
    print("| model | weights | multiply-adds | train | test |")
    print("|---|---|---|---|---|")
    for name, (weights, madds, scores) in rows.items():
        cells = []
        for j in (0, 1):
            vals = [s[j] for s in scores]
            cells.append("%s (%s)" % (fmt(kind, sum(vals) / len(vals)),
                                      ", ".join(fmt(kind, v) for v in vals)))
        print("| %s | %d | %d | %s | %s |" % (name, weights, madds, cells[0], cells[1]))
    print(flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--quick"]
    quick = "--quick" in sys.argv
    extra = dict(a.split("=") for a in args[2:])
    task = args[0] if args else "all"
    steps = 200 if quick else int(args[1]) if len(args) > 1 else 3000
    for name in (("xor", "sin", "spirals") if task == "all" else (task,)):
        run(name, steps, layout=extra.get("layout", "one"), ticks=int(extra.get("ticks", 4)),
            ops=int(extra.get("ops", 16)), k=int(extra.get("k", 4)),
            reads=int(extra.get("reads", 2)), gain=float(extra.get("gain", 1)),
            seeds=1 if quick else int(extra.get("seeds", 3)),
            trace=extra.get("trace", "1") == "1")
