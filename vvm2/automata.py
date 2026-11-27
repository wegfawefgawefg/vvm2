"""Cellular automata: a rule learned on small boards and run on larger ones.

  python -m vvm2.automata line  [STEPS]   one row -> the next row under a one-line rule
  python -m vvm2.automata rules [STEPS]   the same, the rule's table given beside the row
  python -m vvm2.automata life  [STEPS]   Conway's game on a grid
  python -m vvm2.automata quick           every task for a few steps, to see that it runs
  options after STEPS:  rule=110  the one-line rule         k=1      apply the rule k times
                        edge=wall or ring                   reads=3  read pointers (2 or 3)
                        ops=128   the size of the bank      seed=0   base=0  skip the nets
                        tpc=2     ticks an output slot      sizes=12,24,48,96  test sizes
                        whole=0.5 the share of training batches run with whole steps

A cell is dead or live. Each state, and each mark below, is a fixed vector of 16 numbers
from the world's code book; the machine never learns the book. The answer in an output slot
is whichever of dead, live, left mark, right mark its vector is nearest.

line, edge=wall (the default): edges are fixed zero. The input region is a left mark, the
  row, a right mark; a mark counts as a dead cell and stays a mark. The output region has
  as many slots and must hold the left mark, the next row, the right mark. Pointers step
  along a line and stop at its ends. Only the cells are scored.
line, edge=ring: edges wrap. The input region is the row alone, and the device's steps make
  it a ring (the step on from the last cell leads to the first). The output region is a
  line with one spare slot at each end of the row, which the world does not read: a write
  pointer that has finished, either way, has somewhere harmless to stand.
line, k=2 or more: the answer is the row after k applications. Scratch is as long as the
  row plus four, and the machine gets k times the ticks.
rules: the row with marks, then the rule's table as 8 pairs of slots (a key for one of the
  8 neighbourhoods, then that neighbourhood's result as a dead or live cell), the pairs in
  a fresh order every board. The row and the table are two separate lines of the device.
  Some rules are never trained on.
life: the board with a border of marks (left mark down the left side, right mark on the
  other three sides), one cell a slot, row after row. The device's four steps are right,
  left, down, up. Right and left go in reading order: right from the end of a row leads to
  the start of the next one. Down and up stop at the border. The output region is a line:
  the rows of the board in reading order, each with its two marks, without the top and
  bottom border rows. Cells past the border are dead.

Ticks: `tpc` an output slot plus 4 (3 a slot plus 6 for rules and life). Trained on small
boards, tested on larger. The machine is also run with whole steps (core.hard = True), and
a trace is printed. Beside it: a GRU reading the slots both ways and answering at each
slot, once with the machine's weight count and once with its multiply-adds; and a small
convolutional net, which has the rule's locality built in (for rules: a small net given
the table and one neighbourhood at a time).
"""
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import params
from .core import SLOT, Core
from .tape import codebook

DEAD, LIVE, LEFT, RIGHT = 0, 1, 2, 3
KINDS = 4           # what an output slot can be read as
KEY0 = 4            # symbols 4..11: the keys of the eight neighbourhoods
CHARS = ".#[]abcdefgh"


def line_step(x, table, ring):
    """x [B, n] of 0 and 1; table [B, 8] or [8], entry 4 left + 2 centre + right."""
    if ring:
        left, right = x.roll(1, 1), x.roll(-1, 1)
    else:
        pad = F.pad(x, (1, 1))
        left, right = pad[:, :-2], pad[:, 2:]
    return table.expand(len(x), 8).gather(1, 4 * left + 2 * x + right)


def life_step(x):
    """x [B, H, W] of 0 and 1, dead beyond the edge."""
    p = F.pad(x, (1, 1, 1, 1))
    h, w = x.shape[1:]
    near = sum(p[:, a:a + h, b:b + w] for a in range(3) for b in range(3)) - x
    return ((near == 3) | ((near == 2) & (x == 1))).long()


def marked(x):
    """[B, n] cells -> [B, n + 2]: a left mark, the cells, a right mark."""
    return F.pad(F.pad(x, (1, 0), value=LEFT), (0, 1), value=RIGHT)


def line_world(n, width, g, rule=110, k=1, ring=False, tpc=2):
    x = torch.randint(2, (n, width), generator=g)
    table = torch.tensor([(rule >> i) & 1 for i in range(8)])
    y = x
    for _ in range(k):
        y = line_step(y, table, ring)
    if ring:
        place = torch.arange(width)
        w = SimpleNamespace(tokens=x, want=F.pad(y, (1, 1)), src=F.pad(place, (1, 1)),
                            go=torch.stack([place.roll(-1), place.roll(1)]),
                            scored=F.pad(torch.ones(width), (1, 1)).bool())
        w.trained = w.scored
    else:
        w = SimpleNamespace(tokens=marked(x), want=marked(y), src=torch.arange(width + 2),
                            go=None, scored=marked(torch.ones(1, width).long())[0] == 1)
        w.trained = torch.ones(width + 2).bool()
    w.n_out = w.want.shape[1]
    w.ticks = k * (tpc * w.n_out + 4)
    w.scratch = w.n_out + 4 if k > 1 else None
    w.shape, w.steps = "ring" if ring else "line", ["next", "prev"]
    return w


def rules_world(n, width, g, pool, tpc=3):
    x = torch.randint(2, (n, width), generator=g)
    rules = pool[torch.randint(len(pool), (n,), generator=g)]
    table = (rules[:, None] >> torch.arange(8)) & 1
    order = torch.rand(n, 8, generator=g).argsort(dim=1)
    pairs = torch.stack([KEY0 + order, table.gather(1, order)], dim=2).flatten(1)
    row = width + 2
    place = torch.arange(row + 16)
    nxt, prv = place + 1, place - 1
    nxt[row - 1], nxt[-1], prv[0], prv[row] = row - 1, row + 15, 0, row
    w = SimpleNamespace(tokens=torch.cat([marked(x), pairs], dim=1), table=table,
                        want=marked(line_step(x, table, False)), src=torch.arange(row),
                        go=torch.stack([nxt, prv]), trained=torch.ones(row).bool(),
                        scored=marked(torch.ones(1, width).long())[0] == 1, n_out=row,
                        ticks=tpc * row + 6, scratch=None, shape="rules",
                        steps=["next", "prev"])
    return w


def life_world(n, size, g, tpc=3):
    live = torch.rand(n, 1, 1, generator=g) * 0.45 + 0.15     # share of live cells, by board
    x = (torch.rand(n, size, size, generator=g) < live).long()

    def framed(b):
        b = F.pad(b, (1, 1, 1, 1), value=RIGHT)
        b[:, :, 0] = LEFT
        return b

    wide = size + 2
    place = torch.arange(wide * wide)
    go = torch.stack([(place + 1).clamp(max=wide * wide - 1), (place - 1).clamp(min=0),
                      torch.where(place + wide < wide * wide, place + wide, place),
                      torch.where(place >= wide, place - wide, place)])
    cells = framed(torch.ones(1, size, size).long())[:, 1:-1].flatten() == 1
    return SimpleNamespace(tokens=framed(x).flatten(1), want=framed(life_step(x))[:, 1:-1].flatten(1),
                           src=place[wide:-wide], go=go, scored=cells,
                           trained=torch.ones(size * wide).bool(), n_out=size * wide,
                           ticks=tpc * size * wide + 6, scratch=None, shape="grid",
                           side=wide, steps=["right", "left", "down", "up"])


class AutoVM(nn.Module):
    """The machine with the world's code book around it. Nothing here is learned except
    the machine itself and one number, how sharply an answer is read."""

    def __init__(self, **core):
        super().__init__()
        self.core = Core(**core)
        self.register_buffer("book", codebook(KEY0 + 8, seed=7))
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, w, trace=False):
        if w.scratch is not None:
            self.core.scratch = w.scratch
        outs = self.core(self.book[w.tokens], w.ticks, w.n_out, trace=trace, go=w.go)
        return outs[-1] @ self.book[:KINDS].T * self.temp

    def madds(self, w):
        """Multiply-adds for one board: fetch, execute and the seek match, every tick. A
        load is a lookup once pointers are exact and is not counted."""
        c = self.core
        slots = 1 + w.tokens.shape[1] + c.scratch + w.n_out
        tick = c.ops * SLOT + c.k * c.fan_out * (c.reads + 2) * SLOT + (c.reads + 1) * slots * SLOT
        return w.ticks * tick


class TagGRU(nn.Module):
    """Reads the input slots in order, both ways, and answers at each slot."""

    def __init__(self, width):
        super().__init__()
        self.embed = nn.Embedding(KEY0 + 8, width)
        self.gru = nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * width, KINDS)

    def forward(self, w):
        y, _ = self.gru(self.embed(w.tokens))
        return self.out(y[:, w.src])

    def madds(self, w):
        return w.tokens.shape[1] * (params(self.gru) + params(self.out))


class Local(nn.Module):
    """A small net with the neighbourhood built in. line: k layers of 3-wide convolution.
    grid: one 3 by 3 convolution, then two layers a cell. rules: a net a cell, given the
    cell's neighbourhood and the 8 entries of the table."""

    def __init__(self, shape, k=1, width=16):
        super().__init__()
        self.shape = shape
        if shape == "grid":
            self.net = nn.Sequential(nn.Conv2d(KINDS, width, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(width, width, 1), nn.ReLU(),
                                     nn.Conv2d(width, KINDS, 1))
        elif shape == "rules":
            self.net = nn.Sequential(nn.Linear(3 * KINDS + 8, 2 * width), nn.ReLU(),
                                     nn.Linear(2 * width, 2 * width), nn.ReLU(),
                                     nn.Linear(2 * width, KINDS))
        else:
            mode = "circular" if shape == "ring" else "zeros"
            layers, last = [], KINDS
            for _ in range(k):
                layers += [nn.Conv1d(last, width, 3, padding=1, padding_mode=mode), nn.ReLU()]
                last = width
            self.net = nn.Sequential(*layers, nn.Conv1d(width, KINDS, 1))

    def forward(self, w):
        if self.shape == "rules":
            row = F.one_hot(w.tokens[:, :w.n_out], KINDS).float()
            near = F.pad(row, (0, 0, 1, 1)).unfold(1, 3, 1).flatten(2)
            return self.net(torch.cat([near, w.table[:, None].expand(-1, w.n_out, -1).float()], 2))
        x = F.one_hot(w.tokens, KINDS).float().transpose(1, 2)
        if self.shape == "grid":
            x = x.unflatten(2, (w.side, w.side))
        return self.net(x).flatten(2).transpose(1, 2)[:, w.src]

    def madds(self, w):
        return params(self) * (w.n_out if self.shape == "rules" else w.tokens.shape[1])


def gru_width(target, cost):
    """The GRU width whose cost (a function of the model) is nearest the target."""
    return min(range(4, 600, 4), key=lambda width: abs(cost(TagGRU(width)) - target))


def fit(model, make, steps, seed, batch=64, say=0, whole=0.0):
    """say: print the loss to stderr every so many steps. whole: the share of training
    batches on which the machine runs with whole steps (every pointer takes its likeliest
    move outright; the gradient then reaches what is stored and the pc, not the moves)."""
    coin = torch.Generator().manual_seed(seed + 77)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        model.train()
        if whole and torch.rand((), generator=coin) < whole:
            model.eval()
            model.core.hard = True
        w = make(batch, None, g)
        loss = F.cross_entropy(model(w)[:, w.trained].flatten(0, 1), w.want[:, w.trained].flatten())
        if say and step % say == say - 1:
            print("  step %d: loss %.3f" % (step + 1, loss.item()), file=sys.stderr, flush=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if whole:
            model.core.hard = False
    model.eval()


def cells(model, make, sizes, g, boards=300):
    """For each size: share of cells right / share of whole boards right."""
    row = []
    with torch.no_grad():
        for size in sizes:
            w = make(boards, size, g)
            hit = (model(w).argmax(dim=-1) == w.want)[:, w.scored]
            row.append("%.3f / %.2f" % (hit.float().mean(), hit.all(dim=1).float().mean()))
    return " | ".join(row)


def bank_use(core):
    """Of the vop fetched nearest on each tick since the counts were last cleared."""
    q = core.picks / core.picks.sum().clamp_min(1)
    entropy = -(q[q > 0] * q[q > 0].log()).sum() / torch.log(torch.tensor(float(core.ops)))
    return "%d of %d vops ever nearest (%.2f), entropy of use %.2f, the top vop's share %.2f" % (
        (q > 0).sum(), core.ops, (q > 0).float().mean(), entropy, q.max())


def show_trace(vm, w, most=40):
    """One board run with whole steps, tick by tick: the vop nearest the pc, the slot each
    pointer is on for that tick's load and store, and the move it then makes. Places: pc;
    i3 the fourth input slot; s0 the first scratch slot; o2 the third output slot."""
    core = vm.core
    with torch.no_grad():
        got = vm(w, trace=True).argmax(dim=-1)
    log, n_in = core.trace, w.tokens.shape[1]

    def name(i):
        if i == 0:
            return "pc"
        if i <= n_in:
            return "i%d" % (i - 1)
        if i <= n_in + core.scratch:
            return "s%d" % (i - 1 - n_in)
        return "o%d" % (i - 1 - n_in - core.scratch)

    runs = torch.stack([t[0] for t in log], dim=1)                      # [boards, ticks]
    common = runs.mode(dim=0).values
    top = torch.stack([(r == r.mode().values).float().mean() for r in runs]).mean()
    kinds = torch.tensor([len(set(r.tolist())) for r in runs]).float().mean()
    print("  trace, %d input slots, %d ticks, run with whole steps. Over %d boards: %.1f "
          "different vops a run; the most used vop of a run has %.2f of its ticks; on %.3f "
          "of ticks a board's vop is not the commonest one for that tick (how much the "
          "picks change with the input)" % (n_in, len(log), len(runs), kinds, top,
                                            (runs != common).float().mean()))
    print("    in:     " + "".join(CHARS[int(c)] for c in w.tokens[0]))
    print("    out:    " + "".join(CHARS[int(c)] for c in got[0]))
    print("    wanted: " + "".join(CHARS[int(c)] for c in w.want[0]))
    did = ["stay"] + w.steps + ["in0", "inN", "sc0", "scN", "out0", "outN", "seek"]
    ends = [1, n_in, 1 + n_in, n_in + core.scratch, 1 + n_in + core.scratch,
            n_in + core.scratch + w.n_out]
    boot = [ends[int(a)] for a in core.boot_at.argmax(dim=-1)]     # where each pointer starts
    log = log[:most]
    print("    vop:    " + " ".join("%5d" % int(t[0][0]) for t in log)
          + ("  ... (first %d ticks)" % most if len(core.trace) > most else ""))
    for p in range(core.reads + 1):
        on = [boot[p]] + [int(t[1][0, p].argmax()) for t in log[:-1]]
        print("    " + ("write:  " if p == core.reads else "read %d: " % (p + 1))
              + " ".join("%5s" % name(i) for i in on))
        print("      then: " + " ".join("%5s" % did[int(t[2][0, p].argmax())] for t in log))
    sys.stdout.flush()


def rule_pools(seed=3, held=64):
    """Rules never trained on (110, 30, 90, 184 among them) and the rest."""
    order = torch.randperm(256, generator=torch.Generator().manual_seed(seed)).tolist()
    out = [110, 30, 90, 184] + [r for r in order if r not in (110, 30, 90, 184)][:held - 4]
    return torch.tensor([r for r in range(256) if r not in out]), torch.tensor(out)


def run(task, steps, seed=0, reads=3, ops=128, rule=110, k=1, edge="wall", tpc=None,
        base=True, sizes=None, boards=300, whole=0.0):
    torch.set_num_threads(1)     # one thread a job: quicker than four when the CPU is shared
    torch.manual_seed(seed)
    seen, held = rule_pools()
    if task == "life":
        short, tests, tpc = (4, 6), sizes or (6, 8, 12), tpc or 3
        world = lambda n, size, g: life_world(n, size, g, tpc)
    elif task == "rules":
        short, tests, tpc = (6, 12), sizes or (12, 24, 48), tpc or 3
        world = lambda n, size, g: rules_world(n, size, g, held, tpc)
    else:
        short, tests, tpc = (6, 12), sizes or (12, 24, 48, 96), tpc or 2
        world = lambda n, size, g: line_world(n, size, g, rule, k, edge == "ring", tpc)

    def make(n, size, g):       # no size: a training batch (for rules: of trained rules)
        if size is not None:
            return world(n, size, g)
        size = int(torch.randint(short[0], short[1] + 1, (1,), generator=g))
        return rules_world(n, size, g, seen, tpc) if task == "rules" else world(n, size, g)

    g = torch.Generator().manual_seed(1000 + seed)
    big = make(2, short[1], g)
    vm = AutoVM(ops=ops, k=min(4, ops), reads=reads, moves=len(big.steps),
                scratch=big.scratch or 32)
    models = [("vvm, %d read pointers" % reads, vm)]
    if base:
        models += [("local net", Local(big.shape, k)),
                   ("gru, the vvm's multiply-adds", TagGRU(gru_width(vm.madds(big), lambda m: m.madds(big)))),
                   ("gru, the vvm's weights", TagGRU(gru_width(params(vm), params)))]
    what = {"line": "rule %d applied %d time%s, edge %s" % (rule, k, "" if k == 1 else "s", edge),
            "rules": "the rule given as a table; tested on %d rules never trained on" % len(held),
            "life": "Conway's game"}[task]
    print("%s: %s. Trained on sizes %d to %d, %d steps, seed %d, a bank of %d vops, %d ticks "
          "an output slot%s. In each cell: share of cells right / share of whole boards right"
          % (task, what, short[0], short[1], steps, seed, ops, tpc,
             ", %.2f of training batches run with whole steps" % whole if whole else ""))
    with torch.no_grad():
        dead = " | ".join("%.3f" % (w.want[:, w.scored] == DEAD).float().mean()
                          for w in (make(boards, size, g) for size in tests))
    print("| model | weights | multiply-adds a board of size %d | " % short[1]
          + " | ".join("size %d" % t for t in tests) + " |")
    print("|---|---|---|" + "---|" * len(tests))
    print("| answering dead everywhere | 0 | 0 | %s |" % dead, flush=True)
    for name, model in models:
        fit(model, make, steps, seed, *((64, 500, whole) if model is vm else ()))
        size = "%d | %d" % (params(model), model.madds(big))
        print("| %s | %s | %s |" % (name, size, cells(model, make, tests, g, boards)), flush=True)
        if model is vm:
            vm.core.hard = True
            vm.core.picks.zero_()
            print("| the same vvm, run with whole steps | %s | %s |" % (
                size, cells(vm, make, tests, g, boards)), flush=True)
            if task == "rules":
                print("| the same, on the trained rules | %s | %s |" % (size, cells(
                    vm, lambda n, s, g: rules_world(n, s, g, seen, tpc), tests, g, boards)), flush=True)
            print("  bank use, whole steps, all test sizes: " + bank_use(vm.core))
            show_trace(vm, make(200, short[0] if task == "life" else 8, g))
            vm.core.hard = False


def quick():
    """Every task for a few steps: checks that it runs, not that it learns."""
    for task, more in (("line", {}), ("line", {"edge": "ring", "reads": 2}), ("line", {"k": 2}),
                       ("rules", {"reads": 4}), ("life", {})):
        run(task, 20, sizes=(6,) if task == "life" else (12, 24), boards=20, ops=16, **more)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("quick", "--quick"):
        quick()
        sys.exit()
    extra = dict(a.split("=") for a in sys.argv[3:])
    sizes = tuple(int(s) for s in extra["sizes"].split(",")) if "sizes" in extra else None
    run(sys.argv[1] if len(sys.argv) > 1 else "line",
        int(sys.argv[2]) if len(sys.argv) > 2 else 3000, seed=int(extra.get("seed", 0)),
        reads=int(extra.get("reads", 3)), ops=int(extra.get("ops", 128)),
        rule=int(extra.get("rule", 110)), k=int(extra.get("k", 1)),
        edge=extra.get("edge", "wall"), tpc=int(extra["tpc"]) if "tpc" in extra else None,
        base=extra.get("base", "1") == "1", sizes=sizes, whole=float(extra.get("whole", 0)))
