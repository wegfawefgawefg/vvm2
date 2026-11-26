"""Halting: the vvm says when it is done.

  python -m vvm2.halt [TASK OPS TICKS RANK APPLY]     default mnist 16 6 0 ff

Two changes to training, nothing else:
  1. the answer is asked for after every tick, not only the last, so the state can be read
     at any tick (loss: the mean over ticks of the usual loss);
  2. one extra output read from the state, "done": trained to say whether the answer read
     at this tick is right. It is the vvm's own claim that it has finished.
Running: tick until "done" passes a bar (0.5, 0.9, 0.99), at most TICKS ticks, and take the
answer of that tick. Reported: the score and the mean number of ticks at each bar, the score
if stopped at each fixed tick, and how many ticks each class needs.
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classic import TASKS, shifted
from .vop import VopVVM


def run_all(model, x):
    """Answers and "done" after every tick: logits [T, B, C], done [T, B], and the
    best-matching vop of every tick [B, T]."""
    state = F.layer_norm(model.proj(x), (model.d,))
    logits, done, tops = [], [], []
    for _ in range(model.ticks):
        out = model.bank(state)
        tops.append(model.bank.last[:, 0])
        state = F.layer_norm(state + out if model.apply == "res" else out, (model.d,))
        logits.append(model.readout(state))
        done.append(model.halt(state).squeeze(-1))
    return torch.stack(logits), torch.stack(done), torch.stack(tops, dim=1)


def train_every(model, data, steps, batch=128, lr=3e-3, seed=0, wd=0.05, shift=2):
    """Train with the answer and "done" asked for after every tick."""
    x, y = data[0], data[1]
    if not hasattr(model, "halt"):
        model.halt = nn.Linear(model.d, 1)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        model.train()
        i = torch.randint(len(x), (batch,), generator=g)
        xb = shifted(x[i], shift, g) if shift else x[i]
        logits, done, _ = run_all(model, xb)
        ticks = len(logits)
        target = y[i].repeat(ticks)
        loss = F.cross_entropy(logits.flatten(0, 1), target)
        right = (logits.argmax(dim=-1) == y[i]).float().detach()
        loss = loss + F.binary_cross_entropy_with_logits(done, right)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


def halting(logits, done, bar):
    """The tick each example stops at (first tick whose "done" passes the bar, else the
    last) and its answer there."""
    ticks = len(logits)
    passed = torch.sigmoid(done) > bar                      # [T, B]
    first = torch.where(passed.any(dim=0), passed.float().argmax(dim=0),
                        torch.full((passed.shape[1],), ticks - 1))
    answer = logits.argmax(dim=-1).gather(0, first.unsqueeze(0)).squeeze(0)
    return first + 1, answer


def main():
    arg = sys.argv[1:] + ["mnist", "16", "6", "0", "ff"][len(sys.argv) - 1:]
    task, ops, ticks, rank, apply = arg[0], int(arg[1]), int(arg[2]), int(arg[3]), arg[4]
    torch.set_num_threads(8)
    data = TASKS[task](0)
    torch.manual_seed(0)
    model = VopVVM(784, 10, d=64, ticks=ticks, ops=ops, k=2, rank=rank, apply=apply)
    train_every(model, data, 8000)
    model.eval()
    x, y = data[2], data[3]
    with torch.no_grad():
        logits, done, tops = run_all(model, x)
    print("%s, %d vops%s, 2 fetched, at most %d ticks, apply %s" % (
        task, ops, " (rank %d)" % rank if rank else "", ticks, apply))
    print("  right if always stopped at tick 1..%d: %s" % (ticks, " ".join(
        "%.3f" % (logits[t].argmax(dim=1) == y).float().mean() for t in range(ticks))))
    for bar in (0.5, 0.9, 0.99):
        at, answer = halting(logits, done, bar)
        acc = (answer == y).float().mean().item()
        spread = " ".join("%d:%.0f%%" % (t, 100 * (at == t).float().mean())
                          for t in range(1, ticks + 1))
        print("  stop when done > %.2f: right %.3f, mean ticks %.2f   stopped at tick  %s"
              % (bar, acc, at.float().mean().item(), spread))
    at, answer = halting(logits, done, 0.9)
    print("  ticks each class needs (done > 0.9): " + "  ".join(
        "%d: %.2f" % (c, at[y == c].float().mean()) for c in range(10)))
    early = at == 1
    print("  of those stopped at tick 1: right %.3f; of those that ran on: right %.3f" % (
        (answer[early] == y[early]).float().mean(), (answer[~early] == y[~early]).float().mean()))
    paths = [len(set(r[:n].tolist())) for r, n in zip(tops, at.tolist())]
    print("  different vops on a walk (best match per tick, up to its stop): mean %.2f" %
          (sum(paths) / len(paths)))


if __name__ == "__main__":
    main()
