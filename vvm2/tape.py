"""Tasks where steps and pointers are the whole job, tested on longer cases than trained.

  python -m vvm2.tape copy    [STEPS]     copy a list from the input slots to the output slots
  python -m vvm2.tape reverse [STEPS]     the same, back to front
  python -m vvm2.tape simon   [STEPS]     the lights game: watch a sequence of four colours,
                                          one light at a time, then play it back

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

    def lists(self, tokens):
        """tokens [B, L] -> logits [B, L, SYMBOLS] read from the output slots at the end."""
        length = tokens.shape[1]
        outs = self.core(self.book[tokens], 2 * length + 4, line_pos(length), line_pos(length))
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


def run(task, steps):
    torch.set_num_threads(8)
    simon = task == "simon"
    kinds = COLOURS if simon else SYMBOLS
    short, tests = ((1, 6), (4, 6, 8, 10, 12)) if simon else ((2, 8), (8, 12, 16, 24))

    def answer(model, x):
        logits = model.simon(x) if simon else model.lists(x)
        return logits, (x.flip(1) if task == "reverse" else x)

    torch.manual_seed(0)
    vm = TapeVM()
    make = (lambda w: SimonGRU(w)) if simon else (lambda w: ListGRU(w))
    gru = make(sized(make, params(vm)))
    print("%s: trained on lengths %d to %d, %d steps. In each cell: share of items right / "
          "share of whole sequences right" % (task, short[0], short[1], steps))
    print("| model | weights | " + " | ".join("length %d" % t for t in tests) + " |")
    print("|---|---|" + "---|" * len(tests), flush=True)
    for name, model in (("vvm", vm), ("gru", gru)):
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        g = torch.Generator().manual_seed(0)
        for _ in range(steps):
            model.train()
            length = int(torch.randint(short[0], short[1] + 1, (1,), generator=g))
            x = torch.randint(kinds, (64, length), generator=g)
            logits, want = answer(model, x)
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


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "copy",
        int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
