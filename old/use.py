"""How unevenly are the vops used? Is it Zipf-like?

  python -m old.use

Trains a few vvms on MNIST (6,000 steps, weight decay 0.05), counts every fetch over the
10,000 test digits, and plots each vop's share of the fetches from the busiest down, on
log-log axes beside a 1/rank line (Zipf), into docs/img/use.png. Prints the shares and the
slope of a straight-line fit on the log-log plot (Zipf is a slope near -1).
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .classic import TASKS, train
from .vop import VopVVM

SETUPS = [  # label, ops, fetched, ticks, rank
    ("16 full vops, 3 ticks", 16, 2, 3, 0),
    ("64 small vops, 1 tick", 64, 2, 1, 2),
    ("256 small vops, 1 tick", 256, 2, 1, 2),
    ("1,024 small vops, 1 tick", 1024, 2, 1, 2),
]
COLORS = ["#eb6834", "#2a78d6", "#1baf6b", "#7a4fd6"]


def main():
    torch.set_num_threads(8)
    data = TASKS["mnist"](0)
    fig, ax = plt.subplots(figsize=(8.5, 5.6), facecolor="white")
    for (label, ops, k, ticks, rank), color in zip(SETUPS, COLORS):
        torch.manual_seed(0)
        model = VopVVM(784, 10, d=64, ticks=ticks, ops=ops, k=k, rank=rank)
        _, test, _ = train(model, data, 6000, 128, 3e-3, 0, wd=0.05)
        model.bank.picks.zero_()
        model.eval()
        with torch.no_grad():
            model(data[2])
        share = model.bank.picks.sort(descending=True).values
        share = share / share.sum()
        used = share[share > 0]
        n = len(used)
        xs = torch.arange(1, n + 1).float().log()
        ys = used.log()
        fit = min(n, max(4, int((used > 0.002).sum())))  # fit the part above 0.2%
        slope = ((xs[:fit] - xs[:fit].mean()) * (ys[:fit] - ys[:fit].mean())).sum() / \
            ((xs[:fit] - xs[:fit].mean()) ** 2).sum()
        print("%-26s test %.3f  used %4d of %4d  busiest %.0f%%  top-4 %.0f%%  slope %.2f"
              % (label, test, n, ops, 100 * used[0], 100 * used[:4].sum(), slope.item()))
        print("    shares %%: " + " ".join("%.1f" % (100 * v) for v in used[:20].tolist()))
        ax.plot(range(1, n + 1), used.numpy(), marker="o", ms=3.5, lw=1.6, color=color,
                label="%s  (slope %.1f, %d used)" % (label, slope.item(), n))
    ref = [1, 300]
    ax.plot(ref, [0.5 / r for r in ref], color="#888888", lw=1, ls="--", label="1 / rank (Zipf)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 1)
    ax.set_xlabel("vop, from the busiest down")
    ax.set_ylabel("share of all fetches")
    ax.set_title("How unevenly the vops are used (MNIST, all 10,000 test digits)", loc="left",
                 fontsize=12, weight="bold")
    ax.grid(True, which="major", color="#e6e6e6", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("docs/img/use.png", dpi=110)


if __name__ == "__main__":
    main()
