"""What does a class's common path build, vop by vop?

  python -m vvm2.pathlook [TASK OPS TICKS]      default mnist 16 3; TASK: mnist kmnist fashion

Trains one vvm (8,000 steps, weight decay 0.05, training images shifted up to 2 pixels, the
answer asked for after every tick so the state can be read at any tick).
For each class it takes the commonest path (the best-matching vop of each tick, in order)
and the test examples on it, and draws one row into docs/img/paths-TASK.png:

  the average example on the path;
  then, after the input is written in and after each vop of the path:
    used     which pixels that vop used to move the answer toward the right class (red) or
             away from it (blue): the change that tick made to the right class's score,
             traced back to the pixels (gradient times input, averaged)
    holds    the state turned back into a picture by a straight-line decoder fitted for
             that tick: what an outside reader can still see of the image in the state
    answer   what the readout would say if the walk stopped there (the right class dark)

The "written in" stage is before any vop has run; the readout was never trained on it, so
its answer is only a reference.
"""
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .classic import TASKS, score
from .halt import train_every
from .vop import VopVVM


def states_of(model, x):
    """The state after the input write and after every tick, and the best vop per tick."""
    out = [F.layer_norm(model.proj(x), (model.d,))]
    tops = []
    for _ in range(model.ticks):
        nxt = model.bank(out[-1])
        tops.append(model.bank.last[:, 0])
        out.append(F.layer_norm(out[-1] + nxt if model.apply == "res" else nxt, (model.d,)))
    return out, torch.stack(tops, dim=1)


def main():
    arg = sys.argv[1:] + ["mnist", "16", "3"][len(sys.argv) - 1:]
    task, ops, ticks = arg[0], int(arg[1]), int(arg[2])
    torch.set_num_threads(8)
    data = TASKS[task](0)
    torch.manual_seed(0)
    model = VopVVM(784, 10, d=64, ticks=ticks, ops=ops, k=2)
    train_every(model, data, 8000)  # the answer is asked for after every tick (vvm2/halt.py)
    x, y = data[2], data[3]
    fit, test = score(model, data[0][:10000], data[1][:10000], "class"), score(model, x, y, "class")
    model.eval()
    with torch.no_grad():
        # a straight-line decoder from the state back to pixels, one per stage (ridge)
        tr = data[0][:20000]
        mean = tr.mean(dim=0)
        dec = []
        for s in states_of(model, tr)[0]:
            gram = s.T @ s + 1.0 * torch.eye(model.d)
            dec.append(torch.linalg.solve(gram, s.T @ (tr - mean)))
        sts, tops = states_of(model, x)
        right_at = [(model.readout(s).argmax(dim=1) == y).float().mean().item() for s in sts]
    print("%s, %d vops, 2 fetched, %d ticks: train %.3f, test %.3f" % (task, ops, ticks, fit, test))
    print("  right if stopped after the input write, then after each tick: " +
          " ".join("%.3f" % v for v in right_at))
    cols = 1 + 2 + 3 * ticks
    fig, axes = plt.subplots(10, cols, figsize=(1.45 * cols, 15.2), facecolor="white",
                             gridspec_kw={"width_ratios": [1, 1, 1.25] + [1, 1, 1.25] * ticks})
    for d in range(10):
        m = y == d
        path, count = Counter(map(tuple, tops[m].tolist())).most_common(1)[0]
        on = m & (tops == torch.tensor(path)).all(dim=1)
        xs = x[on][:400].clone().requires_grad_(True)
        st, _ = states_of(model, xs)
        scores = [model.readout(s) for s in st]
        beliefs = [torch.softmax(s, dim=1).mean(dim=0).detach() for s in scores]
        used = [None]
        for t in range(1, ticks + 1):
            delta = (scores[t][:, d] - scores[t - 1][:, d]).sum()
            grad = torch.autograd.grad(delta, xs, retain_graph=True)[0]
            used.append((grad * xs).mean(dim=0).detach().view(28, 28))
        holds = [(s.detach().mean(dim=0) @ dec[i] + mean).view(28, 28) for i, s in enumerate(st)]
        row = axes[d]
        row[0].imshow(x[on].mean(dim=0).view(28, 28).numpy(), cmap="gray_r")
        row[0].set_ylabel("%d\n%.0f%% take\n%s" % (d, 100 * count / int(m.sum()),
                          ">".join(map(str, path))), rotation=0, labelpad=34, fontsize=9,
                          va="center")
        col = 1
        for t in range(ticks + 1):
            if t > 0:
                lim = max(used[t].abs().max().item(), 1e-9)
                row[col].imshow(used[t].numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
                row[col].set_title("vop %d used" % path[t - 1], fontsize=8.5, pad=2)
                col += 1
            row[col].imshow(holds[t].numpy(), cmap="gray_r")
            row[col].set_title("holds", fontsize=8.5, pad=2)
            col += 1
            row[col].bar(range(10), beliefs[t].numpy(),
                         color=["#1a1a1a" if c == d else "#b9c4d0" for c in range(10)])
            row[col].set_ylim(0, 1)
            row[col].set_title("answer", fontsize=8.5, pad=2)
            row[col].set_xticks(range(10))
            row[col].tick_params(labelsize=6, length=0)
            row[col].set_yticks([])
            for side in ("top", "right", "left"):
                row[col].spines[side].set_visible(False)
            col += 1
        for ax in row:
            if not ax.patches:  # the image panels
                ax.set_xticks([])
                ax.set_yticks([])
    stages = ["written in"] + ["tick %d" % t for t in range(1, ticks + 1)]
    fig.suptitle("What each class's common path builds, vop by vop  (%s, %d vops, %d ticks, "
                 "test %.3f)\nstages left to right: the average example | %s.   Right if "
                 "stopped at each stage: %s" % (task, ops, ticks, test, " | ".join(stages),
                 ", ".join("%.2f" % v for v in right_at)), fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0.02, 0, 1, 0.965))
    fig.savefig("docs/img/paths-%s.png" % task, dpi=100)


if __name__ == "__main__":
    main()
