"""A ladder of small algorithms on the frozen core, each trained on short inputs and tested
on much longer ones, to find where the machine stops.

  python -m vvm2.algos TASK [STEPS] [options]

  shift     every item moved one place toward the front; the last item stays where it was
  double    every item written twice in a row (a b c -> a a b b c c)
  twice     the whole list, then the whole list again (a b c -> a b c a b c); the input ends
            with a mark, since the machine must find where the list ends to start over
  keep      only the items of one kind (symbols 0 to 3 of 8), packed to the front; the world
            reads as many output slots as there are such items
  count     how many items are of that kind, as a tally: that many output slots marked and
            the rest left empty (an output slot nothing was written to reads as empty)
  parity    a list of bits, then a mark; one output slot: is the number of ones odd
  compare   list, mark, list, mark; one output slot: are the two lists the same
  addbin    two binary numbers, lowest digit first (number, mark, number, mark); the sum,
            lowest digit first, one digit more than the numbers
  adddec    the same in decimal
  brackets  a string of ( and ), then a mark; one output slot: is it balanced
  sort      a list over 4 symbols, then a mark; the list in order
  cond      one signal slot, then a list: the signal says copy the list or reverse it
  condflash the same, but the signal slot goes dark after the first tick, so the choice
            must be held by the machine

  options:  ops=128   the size of the bank          k=4      vops fetched a tick
            seed=0                                  gru=0    skip the GRUs
            threads=1                               twin=1   the seek-my-twin step
            boot=1    start the read pointers on the first input slot and the write pointer
                      on the first output slot (only where they start; still learned)
            seek=0    hold the moves by content off (tape.no_seek)
            lo=2 hi=8 the trained lengths           tests=8,16,32,64

An item is one of 12 fixed vectors of 16 numbers (the world's code book: ten symbols and two
marks; the machine never learns it). The answer in an output slot is whichever answer
vector it is nearest. The machine gets 2 ticks for each input or output slot, whichever is
more, plus 4 (sort: 4 a slot plus 8). Trained on lengths 2 to 8.

Beside the machine, two GRUs that read the items and are then asked for the answers one by
one: one with the same number of weights, one doing the same arithmetic a step as the
machine does a tick. The machine is also run with whole steps (`core.hard`).
"""
import math
import sys
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import params
from .core import SLOT, Core
from .tape import codebook, no_seek

SYMBOLS = 8
MARK, MARK2, DARK = 10, 11, 12          # two marks, and a slot with nothing in it
NAMES = "0123456789MR_"
BAR = 0.4                               # an output slot less like every answer than this is empty
MOVES = ["  .", "  >", "  <", "in0", "inN", "sc0", "scN", "ou0", "ouN", "seek", "twin"]


def default_ticks(n_in, n_out):
    return 2 * max(n_in, n_out) + 4


@dataclass
class Task:
    make: Callable                      # (n, length, g) -> tokens [B, n_in], want [B, n_out], mask
    answers: list                       # the code book rows an output slot is read against
    empty: bool = False                 # one more answer: nothing written here
    ticks: Callable = default_ticks
    even: bool = False                  # only even lengths
    later: Callable = None              # tokens -> what the input slots show after the first tick


def rand(n, length, g, kinds=SYMBOLS):
    return torch.randint(kinds, (n, length), generator=g)


def mark(n, m=MARK):
    return torch.full((n, 1), m)


def shift(n, length, g):
    x = rand(n, length, g)
    return x, torch.cat([x[:, 1:], x[:, -1:]], dim=1), None


def double(n, length, g):
    x = rand(n, length, g)
    return x, x.repeat_interleave(2, dim=1), None


def twice(n, length, g):
    x = rand(n, length, g)
    return torch.cat([x, mark(n)], dim=1), torch.cat([x, x], dim=1), None


def keep(n, length, g):
    x = rand(n, length, g)
    kept = x < 4
    order = (~kept).long().argsort(dim=1, stable=True)
    return x, x.gather(1, order), kept.gather(1, order)


def count(n, length, g):
    x = rand(n, length, g)
    c = (x < 4).sum(dim=1, keepdim=True)
    return torch.cat([x, mark(n)], dim=1), (torch.arange(length) >= c).long(), None


def parity(n, length, g):
    x = rand(n, length, g, 2)
    return torch.cat([x, mark(n)], dim=1), x.sum(dim=1, keepdim=True) % 2, None


def compare(n, length, g):
    """Half the pairs are the same; in the rest one place differs, and each other place
    differs one time in four."""
    a = rand(n, length, g)
    change = torch.rand(n, length, generator=g) < 0.25
    change[torch.arange(n), torch.randint(length, (n,), generator=g)] = True
    change &= torch.rand(n, 1, generator=g) < 0.5
    other = (a + torch.randint(1, SYMBOLS, (n, length), generator=g)) % SYMBOLS
    b = torch.where(change, other, a)
    same = (~change.any(dim=1, keepdim=True)).long()
    return torch.cat([a, mark(n), b, mark(n)], dim=1), same, None


def adder(base):
    def make(n, length, g):
        a, b = rand(n, length, g, base), rand(n, length, g, base)
        digits, carry = [], torch.zeros(n, dtype=torch.long)
        for i in range(length):
            s = a[:, i] + b[:, i] + carry
            digits.append(s % base)
            carry = s // base
        digits.append(carry)
        return torch.cat([a, mark(n), b, mark(n)], dim=1), torch.stack(digits, dim=1), None
    return make


def brackets(n, length, g):
    """0 is ( and 1 is ). Half are balanced strings left alone; a quarter have one bracket
    turned round (the counts no longer agree); a quarter are shuffled (the counts agree,
    the order usually does not). The answer is worked out from the string itself."""
    s = torch.zeros(n, length, dtype=torch.long)
    depth = torch.zeros(n, dtype=torch.long)
    for i in range(length):
        close = ((torch.rand(n, generator=g) < 0.5) & (depth > 0)) | (depth >= length - i)
        s[:, i] = close.long()
        depth = depth + 1 - 2 * close.long()
    kind = torch.randint(4, (n,), generator=g)
    turned = s.clone()
    rows = torch.arange(n)
    place = torch.randint(length, (n,), generator=g)
    turned[rows, place] = 1 - turned[rows, place]
    mixed = s.gather(1, torch.rand(n, length, generator=g).argsort(dim=1))
    s = torch.where((kind == 2)[:, None], turned, torch.where((kind == 3)[:, None], mixed, s))
    level = (1 - 2 * s).cumsum(dim=1)
    ok = (level.min(dim=1).values >= 0) & (level[:, -1] == 0)
    return torch.cat([s, mark(n)], dim=1), ok.long()[:, None], None


def sort(n, length, g):
    x = rand(n, length, g, 4)
    return torch.cat([x, mark(n)], dim=1), x.sort(dim=1).values, None


def cond(n, length, g):
    """The signal is M for copy, R for reverse."""
    x = rand(n, length, g)
    signal = MARK + torch.randint(2, (n, 1), generator=g)
    return torch.cat([signal, x], dim=1), torch.where(signal == MARK, x, x.flip(1)), None


def dark_signal(tokens):
    later = tokens.clone()
    later[:, 0] = DARK
    return later


ALL = list(range(SYMBOLS))
TASKS = {
    "shift": Task(shift, ALL), "double": Task(double, ALL), "twice": Task(twice, ALL),
    "keep": Task(keep, ALL), "count": Task(count, [MARK], empty=True),
    "parity": Task(parity, [0, 1]), "compare": Task(compare, [0, 1]),
    "addbin": Task(adder(2), [0, 1]), "adddec": Task(adder(10), list(range(10))),
    "brackets": Task(brackets, [0, 1], even=True),
    "sort": Task(sort, [0, 1, 2, 3], ticks=lambda n_in, n_out: 4 * n_in + 8),
    "cond": Task(cond, ALL), "condflash": Task(cond, ALL, later=dark_signal),
}


def book():
    return torch.cat([codebook(12), torch.zeros(1, SLOT)])


class AlgoVM(nn.Module):
    """The machine with the world's code book around it. Nothing is learned but the machine
    and one number, how sharply an answer is read."""

    def __init__(self, **core):
        super().__init__()
        self.core = Core(**core)
        self.register_buffer("book", book())
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, task, tokens, n_out, trace=False):
        first = self.book[tokens]
        world = first
        if task.later is not None:
            after = self.book[task.later(tokens)]
            world = lambda t: first if t == 0 else after
        outs = self.core(world, task.ticks(tokens.shape[1], n_out), n_out, trace=trace)
        logits = outs[-1] @ self.book[task.answers].T * self.temp
        if task.empty:
            logits = torch.cat([logits, (self.temp * BAR).expand(*logits.shape[:2], 1)], dim=2)
        return logits


class AlgoGRU(nn.Module):
    def __init__(self, width, classes):
        super().__init__()
        self.width = width
        self.embed = nn.Embedding(len(NAMES), width)
        self.enc = nn.GRU(width, width, batch_first=True)
        self.dec = nn.GRU(1, width, batch_first=True)
        self.out = nn.Linear(width, classes)

    def forward(self, task, tokens, n_out, trace=False):
        _, h = self.enc(self.embed(tokens))
        y, _ = self.dec(torch.ones(tokens.shape[0], n_out, 1), h)
        return self.out(y)


def vm_madds(core, slots):
    """Multiply-adds a tick: the key match, the k vops, and the two matches by content."""
    ptrs = core.reads + 1
    return core.ops * SLOT + core.k * core.fan_out * (ptrs + 1) * SLOT + 2 * ptrs * slots * SLOT


def gru_madds(width):
    return 6 * width * width            # one GRU step at this width


def scores(model, task, length, g, n=300):
    tokens, want, mask = task.make(n, length, g)
    with torch.no_grad():
        hit = model(task, tokens, want.shape[1]).argmax(dim=-1) == want
    if mask is None:
        return hit.float().mean().item(), hit.all(dim=1).float().mean().item()
    return hit[mask].float().mean().item(), (hit | ~mask).all(dim=1).float().mean().item()


def row(model, task, tests, g):
    return " | ".join("%.2f / %.2f" % scores(model, task, t, g) for t in tests)


def bank_use(core, runs):
    """From the vops picked while testing: share of the bank ever used, entropy of use in
    bits, the top vop's share; and from one batch, the share of ticks at which a run's vop
    is not the one most runs have at that tick (how much the program changes with the input)."""
    p = core.picks / core.picks.sum().clamp_min(1)
    used = (p > 0).float().mean().item()
    entropy = -(p[p > 0] * p[p > 0].log2()).sum().item()
    modal = (runs == runs.mode(dim=0).values).float().mean().item()
    return "bank: %.0f%% of %d vops used, entropy %.2f bits, top vop %.0f%% of ticks; %.0f%% " \
           "of ticks differ from the usual vop for that tick" % (
               100 * used, core.ops, entropy, 100 * p.max().item(), 100 * (1 - modal))


def place(i, n_in, scratch):
    if i == 0:
        return "pc"
    if i <= n_in:
        return "i%d" % (i - 1)
    if i <= n_in + scratch:
        return "s%d" % (i - 1 - n_in)
    return "o%d" % (i - 1 - n_in - scratch)


def show_trace(vm, task, length, g, label):
    """A few runs tick by tick: which vop ran (the best match) and where each pointer is.
    i3 is the fourth input slot, s0 the first scratch slot, o2 the third output slot. One run
    for each first input item or one-slot answer, so a program that changes with it shows."""
    tokens, want, mask = task.make(200, length, g)
    n_out = want.shape[1]
    with torch.no_grad():
        got = vm(task, tokens, n_out, trace=True).argmax(dim=-1)
    core, log = vm.core, vm.core.trace
    runs = torch.stack([t[0] for t in log], dim=1)
    kinds = want[:, 0] if n_out == 1 else tokens[:, 0] * (tokens[:, 0] >= MARK)
    picks = [int((kinds == v).nonzero()[0]) for v in kinds.unique()][:2]
    names = [NAMES[a] for a in task.answers] + ["_"]
    print("  trace, %s, length %d: pointers exact to %.2f; %.1f different vops a run" % (
        label, length, core.sharp(),
        torch.tensor([len(set(r.tolist())) for r in runs]).float().mean()))
    for b in picks:
        shown = range(n_out) if mask is None else mask[b].nonzero().flatten().tolist()
        print("    input %s   wanted %s   got %s" % (
            "".join(NAMES[t] for t in tokens[b]), "".join(names[want[b, i]] for i in shown),
            "".join(names[got[b, i]] for i in shown)))
        print("      vop:    " + " ".join("%3d" % int(t[0][b]) for t in log))
        for p in range(core.reads + 1):
            who = "write:  " if p == core.reads else "read %d: " % (p + 1)
            print("      " + who + " ".join(
                "%3s" % place(int(t[1][b, p].argmax()), tokens.shape[1], core.scratch) for t in log))
            if core.hard:               # the move that took it there
                print("        did:  " + " ".join("%3s" % MOVES[int(t[2][b, p].argmax())] for t in log))
    sys.stdout.flush()
    return runs


def train(model, task, steps, lengths, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        model.train()
        length = lengths[int(torch.randint(len(lengths), (1,), generator=g))]
        tokens, want, mask = task.make(64, length, g)
        logits = model(task, tokens, want.shape[1])
        if mask is not None:
            logits, want = logits[mask], want[mask]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), want.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % max(1, steps // 5) == 0:
            print("    step %d loss %.3f" % (step + 1, loss.item()), file=sys.stderr, flush=True)
    model.eval()
    return g


def run(name, steps, ops=128, k=4, seed=0, gru=True, threads=1, twin=False, boot=False,
        lo=2, hi=8, tests=(8, 16, 32, 64), seek=True):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    task = TASKS[name]
    lengths = [n for n in range(lo, hi + 1) if not (task.even and n % 2)]
    vm = AlgoVM(ops=ops, k=min(k, ops), seek_self=twin)
    if not seek:
        no_seek(vm.core)
    if boot:
        with torch.no_grad():
            vm.core.boot_at[:-1, 0] = 3.0
            vm.core.boot_at[-1, 4] = 3.0
    classes = len(task.answers) + int(task.empty)
    tokens, want, _ = task.make(1, hi, torch.Generator().manual_seed(0))
    slots = 1 + tokens.shape[1] + vm.core.scratch + want.shape[1]
    same = min(range(8, 600, 4), key=lambda w: abs(params(AlgoGRU(w, classes)) - params(vm)))
    small = max(4, round(math.sqrt(vm_madds(vm.core, slots) / 6)))
    models = [("vvm", vm)]
    if gru:
        models += [("gru, same weights", AlgoGRU(same, classes)),
                   ("gru, same arithmetic a step", AlgoGRU(small, classes))]
    print("%s: trained on lengths %d to %d, %d steps, seed %d, a bank of %d vops, %d fetched "
          "a tick%s%s%s. In each cell: share of items right / share of whole answers right" % (
              name, lo, hi, steps, seed, ops, vm.core.k,
              ", with the seek-my-twin step" if twin else "",
              "" if seek else ", moves by content held off",
              ", pointers started on the first input and output slots" if boot else ""))
    print("| model | weights | multiply-adds a tick | " +
          " | ".join("length %d" % t for t in tests) + " |")
    print("|---|---|---|" + "---|" * len(tests), flush=True)
    for label, model in models:
        g = train(model, task, steps, lengths, seed)
        cost = vm_madds(vm.core, slots) if label == "vvm" else gru_madds(model.width)
        if label == "vvm":
            vm.core.picks.zero_()
        print("| %s | %d | %d | %s |" % (label, params(model), cost, row(model, task, tests, g)),
              flush=True)
        if label != "vvm":
            continue
        vm.core.hard = True
        print("| vvm, run with whole steps | %d | %d | %s |" % (
            params(model), cost, row(model, task, tests, g)), flush=True)
        vm.core.hard = False
        runs = show_trace(vm, task, 6, g, "as trained")
        print("  " + bank_use(vm.core, runs))
        vm.core.hard = True
        show_trace(vm, task, 6, g, "run with whole steps")
        show_trace(vm, task, 12, g, "run with whole steps")
        vm.core.hard = False


if __name__ == "__main__":
    extra = dict(a.split("=") for a in sys.argv[3:])
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3000,
        ops=int(extra.get("ops", 128)), k=int(extra.get("k", 4)), seed=int(extra.get("seed", 0)),
        gru=extra.get("gru", "1") == "1", threads=int(extra.get("threads", 1)),
        twin=extra.get("twin", "0") == "1", boot=extra.get("boot", "0") == "1",
        lo=int(extra.get("lo", 2)), hi=int(extra.get("hi", 8)), seek=extra.get("seek", "1") == "1",
        tests=tuple(int(t) for t in extra.get("tests", "8,16,32,64").split(",")))
