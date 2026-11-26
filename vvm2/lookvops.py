"""One box per vop: what each vop looks at and what it does with it.

  python -m vvm2.lookvops [OPS]        default 64; rank-2 vops, 2 fetched, 1 tick

With small vops the bank has to share the work, so each vop is easy to read. A rank-2 vop
reads the state along two directions and writes along two others. Drawn for every vop that
is fetched for at least 0.5% of the test digits, busiest first, into docs/img/vops-small.png:

  sent     the average digit this vop is fetched for, less the average of all digits
  looks 1, looks 2   the two things it reads, as pictures: the pixels that raise (red) or
           lower (blue) each of its two read-outs of the state (through the input write;
           the state's normalisation is left out, so this is close, not exact)
  does     how much applying this vop alone moves the answer toward each digit, on the
           digits it is fetched for

Then, in text, the pair of vops each digit is most often given.
"""
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .classic import TASKS, train
from .vop import VopVVM

INK, MUTED = "#1a1a1a", "#555555"


def main():
    ops = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    torch.set_num_threads(8)
    data = TASKS["mnist"](0)
    torch.manual_seed(0)
    model = VopVVM(784, 10, d=64, ticks=1, ops=ops, k=2, rank=2)
    _, test, _ = train(model, data, 3000, 128, 3e-3, 0)
    x, y = data[2], data[3]
    b = model.bank
    model.eval()
    with torch.no_grad():
        state = F.layer_norm(model.proj(x), (model.d,))
        b(state)
        got = b.last                                    # [n, 2] the two vops fetched
        base, before = x.mean(dim=0), model.readout(state)
        rows = []
        for op, count in Counter(got.flatten().tolist()).most_common():
            m = (got == op).any(dim=1)
            if m.float().mean() < 0.005:
                continue
            v = b.v[op]                                  # [2, d] what it reads
            looks = ((v - v.mean(dim=1, keepdim=True)) @ model.proj.weight).view(2, 28, 28)
            out = F.gelu(state[m] @ v.T @ b.u[op].T + b.b[op])
            push = (model.readout(F.layer_norm(out, (model.d,))) - before[m]).mean(dim=0)
            digits = Counter(y[m].tolist())
            names = " ".join("%d:%d%%" % (d, round(100 * c / int(m.sum())))
                             for d, c in digits.most_common(3) if c / int(m.sum()) >= 0.12)
            rows.append((op, m.float().mean().item(), (x[m].mean(dim=0) - base).view(28, 28),
                         looks, push, names or "mixed"))
    print("mnist, %d rank-2 vops, 2 fetched, 1 tick: test %.3f; %d vops drawn" %
          (ops, test, len(rows)))
    for d in range(10):
        pairs = Counter(tuple(sorted(p)) for p in got[y == d].tolist())
        n = int((y == d).sum())
        print("  digit %d: %d different pairs; commonest %s" % (d, len(pairs), "  ".join(
            "%d+%d %.0f%%" % (a, c, 100 * k / n) for (a, c), k in pairs.most_common(4))))
    cols = 2
    per = (len(rows) + cols - 1) // cols
    fig = plt.figure(figsize=(15, 1.02 * per + 1.5), facecolor="white")
    back = fig.add_axes([0, 0, 1, 1])
    back.set_axis_off()
    h = 0.86 / per
    for j, (op, share, sent, looks, push, names) in enumerate(rows):
        cx = 0.03 + (j // per) * 0.49
        cy = 0.91 - (j % per + 0.5) * h
        back.text(cx, cy, "vop %d\n%.0f%% of digits\n%s" % (op, 100 * share, names),
                  fontsize=8.5, color=INK, va="center")
        for i, (img, title) in enumerate([(sent, "sent"), (looks[0], "looks 1"),
                                          (looks[1], "looks 2")]):
            ax = fig.add_axes([cx + 0.085 + i * 0.062, cy - h * 0.42, 0.055, h * 0.84])
            lim = max(abs(img.min().item()), abs(img.max().item()), 1e-6)
            ax.imshow(img.numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_xticks([])
            ax.set_yticks([])
            if j % per == 0:
                ax.set_title(title, fontsize=9, color=MUTED)
        ax = fig.add_axes([cx + 0.285, cy - h * 0.36, 0.16, h * 0.72])
        ax.bar(range(10), push.numpy(), color=["#eb6834" if p > 0 else "#2a78d6" for p in push])
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.set_xticks(range(10))
        ax.tick_params(labelsize=7, length=0)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        if j % per == 0:
            ax.set_title("does: push on each digit", fontsize=9, color=MUTED)
    back.text(0.03, 0.965, "One box per vop: what it is fetched for, what it looks at, what it "
              "does", fontsize=14, color=INK, weight="bold", transform=fig.transFigure)
    back.text(0.03, 0.94, "MNIST, %d rank-2 vops, 2 fetched, 1 tick, test %.3f. Every vop "
              "fetched for at least 0.5%% of test digits, busiest first." % (ops, test),
              fontsize=10, color=MUTED, transform=fig.transFigure)
    fig.savefig("docs/img/vops-small.png", dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    main()
