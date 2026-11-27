"""Tasks where steps and pointers are the whole job, tested on longer cases than trained.

  python -m vvm2.tape copy    [STEPS]     copy a list from the input slots to the output slots
  python -m vvm2.tape reverse [STEPS]     the same, back to front
  python -m vvm2.tape simon   [STEPS]     the lights game: watch a sequence of four colours,
                                          one light at a time, then play it back
  options after STEPS:  ops=8     the size of the bank (a small one cannot hold a program
                                  written out step by step, so it has to loop)
                        shift=16  lists begin anywhere in the first 16 places of the line
                        gru=0     skip the GRU
                        sharp=20  how sharply a pointer picks slots at the start (it is learned)
                        rates=searched   the second set of wave rates for places on a line

Lists: the world lays the list out one item a slot along a line, and reads the answer from
output slots along a line of the same length. An item is one of 8 symbols, each a fixed
vector of 16 numbers (the world's code book; the machine never learns it). The answer in an
output slot is whichever symbol's vector it is nearest. The machine gets 2 ticks an item
plus 4. Trained on lengths 2 to 8, tested on 8, 12, 16 and 24.

Simon: there is one "light" slot and one "signal" slot, and the world changes them while the
machine runs. Each colour is lit for 2 ticks while the signal says watch; then the light
goes dark, the signal says play, and every 2 ticks the world reads one "button" slot. The
sequence is not in memory unless the machine put it there. Trained on 1 to 6 lights, tested
up to 12.

Beside the machine: a GRU of about the same size (for lists it reads the items, then is
asked for the answers one by one; for Simon it sees the same two slots tick by tick).
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import params
from .core import SLOT, Core, line_pos, rand_pos

SYMBOLS = 8
COLOURS = 4
HOLD = 2


def codebook(n, seed=7):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, SLOT, generator=g), dim=1) * 0.9


class TapeVM(nn.Module):
    """The machine with the world's code book around it. Nothing here is learned except
    the machine itself and one number, how sharply an answer is read."""

    def __init__(self, **core):
        super().__init__()
        self.core = Core(**core)
        self.register_buffer("book", codebook(SYMBOLS + 2))   # 8 symbols, then watch, play
        self.temp = nn.Parameter(torch.tensor(10.0))

    def lists(self, tokens, start=0, trace=False):
        """tokens [B, L] -> logits [B, L, SYMBOLS] read from the output slots at the end.
        start: where along the line the list (and its answer) begins."""
        length = tokens.shape[1]
        outs = self.core(self.book[tokens], 2 * length + 4, line_pos(length, start),
                         line_pos(length, start), trace=trace)
        return outs[-1] @ self.book[:SYMBOLS].T * self.temp

    def simon(self, colours):
        """colours [B, L] in 0..3 -> logits [B, L, COLOURS], one a play step."""
        n, length = colours.shape
        watch, play = self.book[SYMBOLS], self.book[SYMBOLS + 1]

        def world(t):
            if t < HOLD * length:
                light = self.book[colours[:, t // HOLD]]
                signal = watch.expand(n, -1)
            else:
                light = torch.zeros(n, SLOT)
                signal = play.expand(n, -1)
            return torch.stack([light, signal], dim=1)

        outs = self.core(world, 2 * HOLD * length, rand_pos(2, 5), rand_pos(1, 6))
        read = [HOLD * length + HOLD * (i + 1) - 1 for i in range(length)]
        return outs[read, :, 0].transpose(0, 1) @ self.book[:COLOURS].T * self.temp


class ListGRU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embed = nn.Embedding(SYMBOLS, width)
        self.enc = nn.GRU(width, width, batch_first=True)
        self.dec = nn.GRU(1, width, batch_first=True)
        self.out = nn.Linear(width, SYMBOLS)

    def lists(self, tokens):
        _, h = self.enc(self.embed(tokens))
        y, _ = self.dec(torch.ones(tokens.shape[0], tokens.shape[1], 1), h)
        return self.out(y)


class SimonGRU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer("book", codebook(SYMBOLS + 2))
        self.gru = nn.GRU(2 * SLOT, width, batch_first=True)
        self.out = nn.Linear(width, COLOURS)

    def simon(self, colours):
        n, length = colours.shape
        light = torch.cat([self.book[colours].repeat_interleave(HOLD, dim=1),
                           torch.zeros(n, HOLD * length, SLOT)], dim=1)
        signal = torch.cat([self.book[SYMBOLS].expand(n, HOLD * length, -1),
                            self.book[SYMBOLS + 1].expand(n, HOLD * length, -1)], dim=1)
        y, _ = self.gru(torch.cat([light, signal], dim=2))
        read = [HOLD * length + HOLD * (i + 1) - 1 for i in range(length)]
        return self.out(y[:, read])


def sized(make, target):
    """The width at which a model has about `target` weights."""
    return min(range(8, 600, 4), key=lambda w: abs(params(make(w)) - target))


def show_trace(vm, task, length, g):
    """One list, tick by tick: which vop ran (the best match), which input slot the first
    read pointed at most, which output slot the write pointed at most. A loop shows as the
    same few vops coming round again while both pointers step along."""
    x = torch.randint(SYMBOLS, (200, length), generator=g)
    with torch.no_grad():
        vm.lists(x, trace=True)
    core, log = vm.core, vm.core.trace
    first_in, first_out = core.head, core.head + length + core.scratch
    vops = [int(t[0][0]) for t in log]
    reads = [int(t[1][0, 0, first_in:first_in + length].argmax()) for t in log]
    on_in = [float(t[1][0, 0, first_in:first_in + length].sum()) for t in log]
    writes = [int(t[2][0, first_out:first_out + length].argmax()) for t in log]
    on_out = [float(t[2][0, first_out:first_out + length].sum()) for t in log]
    distinct = torch.stack([t[0] for t in log], dim=1)
    kinds = torch.tensor([len(set(r.tolist())) for r in distinct]).float().mean()
    print("  trace, %s of length %d, %d ticks: %.1f different vops a run" % (
        task, length, len(log), kinds))
    print("    vop run:        " + " ".join("%3d" % v for v in vops))
    print("    read points at: " + " ".join("%3s" % (r if m > 0.5 else ".") for r, m in zip(reads, on_in)))
    print("    write points at:" + " ".join("%3s" % (w if m > 0.5 else ".") for w, m in zip(writes, on_out)),
          flush=True)


def run(task, steps, ops=128, shift=0, only_vvm=False, sharp=8.0, rates="first"):
    from . import core
    core.line_rates = rates
    torch.set_num_threads(8)
    simon = task == "simon"
    kinds = COLOURS if simon else SYMBOLS
    short, tests = ((1, 6), (4, 6, 8, 10, 12)) if simon else ((2, 8), (8, 12, 16, 24))

    def answer(model, x, start=0):
        if simon:
            logits = model.simon(x)
        else:
            logits = model.lists(x, start) if isinstance(model, TapeVM) else model.lists(x)
        return logits, (x.flip(1) if task == "reverse" else x)

    torch.manual_seed(0)
    vm = TapeVM(ops=ops, k=min(4, ops), sharp=sharp)
    make = (lambda w: SimonGRU(w)) if simon else (lambda w: ListGRU(w))
    gru = make(sized(make, params(vm)))
    print("%s: trained on lengths %d to %d, %d steps, a bank of %d vops%s. In each cell: "
          "share of items right / share of whole sequences right" % (
              task, short[0], short[1], steps, ops,
              ", lists starting anywhere in the first %d places" % shift if shift else ""))
    print("| model | weights | " + " | ".join("length %d" % t for t in tests) + " |")
    print("|---|---|" + "---|" * len(tests), flush=True)
    for name, model in (("vvm", vm),) if only_vvm else (("vvm", vm), ("gru", gru)):
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        g = torch.Generator().manual_seed(0)
        for _ in range(steps):
            model.train()
            length = int(torch.randint(short[0], short[1] + 1, (1,), generator=g))
            x = torch.randint(kinds, (64, length), generator=g)
            start = int(torch.randint(shift + 1, (1,), generator=g)) if shift else 0
            logits, want = answer(model, x, start)
            loss = F.cross_entropy(logits.reshape(-1, kinds), want.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        cells = []
        with torch.no_grad():
            for length in tests:
                x = torch.randint(kinds, (500, length), generator=g)
                logits, want = answer(model, x)
                hit = logits.argmax(dim=-1) == want
                cells.append("%.2f / %.2f" % (hit.float().mean(), hit.all(dim=1).float().mean()))
        print("| %s | %d | %s |" % (name, params(model), " | ".join(cells)), flush=True)
        if name == "vvm":
            print("  sharpness of the pointers after training: %.1f" % vm.core.sharp.item(),
                  flush=True)
        if name == "vvm" and not simon:
            for length in (8, 12):
                show_trace(vm, task, length, g)


if __name__ == "__main__":
    extra = dict(a.split("=") for a in sys.argv[3:])
    run(sys.argv[1] if len(sys.argv) > 1 else "copy",
        int(sys.argv[2]) if len(sys.argv) > 2 else 4000,
        ops=int(extra.get("ops", 128)), shift=int(extra.get("shift", 0)),
        only_vvm=extra.get("gru", "1") == "0", sharp=float(extra.get("sharp", 8)),
        rates=extra.get("rates", "first"))
