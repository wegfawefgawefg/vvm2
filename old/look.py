"""What is each vop doing on MNIST, and what do the walks add up to?

  python -m old.look [OPS FETCHED TICKS]        default 16 2 3

Trains one vvm, then follows the best-matching vop of every tick for the 10,000 test digits
and draws pictures into docs/img/:

  walk-mnist.png   the walk as a graph: a column per tick, a box per vop in use at that
                   tick showing the average digit sent to it (less the average of all
                   digits), which digits those are, and lines for where its inputs go next
  vops-mnist.png   what each vop does to the answer: for the digits sent to it, how much
                   applying that vop alone moves the readout toward or away from each digit
  digits-mnist.png the paths each digit takes: for every digit, its commonest whole paths,
                   what share of that digit takes each, and the average of the digits on it
  program-mnist.png  the same walks with one box per vop, whatever the tick: an arrow from
                   a vop to the vop that ran after it, a loop where a vop ran twice running.
                   A vop is not a layer: the same vop can run early for one digit and late
                   for another

and prints the same as text, with the commonest whole paths.
"""
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .classic import TASKS, train
from .vop import VopVVM

INK, MUTED = "#1a1a1a", "#555555"


def walk(model, x):
    """Per tick: the state going in, the best-matching vop, and the state coming out."""
    b, d = model.bank, model.d
    state = F.layer_norm(model.proj(x), (d,))
    steps = []
    for _ in range(model.ticks):
        out = b(state)
        top = b.last[:, 0]
        after = F.layer_norm(state + out if model.apply == "res" else out, (d,))
        steps.append((state, top, after))
        state = after
    return steps


def alone(model, state, op):
    """The state after applying one vop by itself."""
    b = model.bank
    out = F.gelu(state @ b.w[op].T + b.b[op])
    return F.layer_norm(state + out if model.apply == "res" else out, (model.d,))


def digits_of(y, least=0.12):
    c = Counter(y.tolist())
    n = len(y)
    top = [(d, k / n) for d, k in c.most_common(4) if k / n >= least]
    return " ".join("%d:%d%%" % (d, round(100 * s)) for d, s in top) or "mixed"


def main():
    ops, k, ticks = (int(v) for v in (sys.argv[1:4] + ["16", "2", "3"][len(sys.argv) - 1:]))
    torch.set_num_threads(8)
    data = TASKS["mnist"](0)
    torch.manual_seed(0)
    model = VopVVM(784, 10, d=64, ticks=ticks, ops=ops, k=k)
    _, test, _ = train(model, data, 3000, 128, 3e-3, 0)
    x, y = data[2], data[3]  # all 10,000 test digits, about 1,000 of each
    model.eval()
    with torch.no_grad():
        steps = walk(model, x)
        base = x.mean(dim=0)
        nodes = []  # per tick: list of (op, share, mean image, digits text, logit change)
        for t, (before, top, _) in enumerate(steps):
            row = []
            for op in sorted(set(top.tolist()), key=lambda o: -(top == o).sum().item()):
                m = top == op
                if m.float().mean() < 0.01:
                    continue
                move = (model.readout(alone(model, before[m], op))
                        - model.readout(before[m])).mean(dim=0)
                row.append((op, m.float().mean().item(), (x[m].mean(dim=0) - base).view(28, 28),
                            digits_of(y[m]), move))
            nodes.append(row)
    print("mnist, %d vops, %d fetched, %d ticks: test %.3f" % (ops, k, ticks, test))
    for t, row in enumerate(nodes):
        print("tick %d" % (t + 1))
        for op, share, _, text, move in row:
            up = move.topk(2).indices.tolist()
            down = (-move).topk(2).indices.tolist()
            print("  vop %2d  %4.0f%% of digits  sent: %-22s  pushes toward %s, away from %s"
                  % (op, 100 * share, text, up, down))
    paths = defaultdict(list)
    tops = torch.stack([s[1] for s in steps], dim=1)
    for p, label in zip(map(tuple, tops.tolist()), y.tolist()):
        paths[p].append(label)
    print("commonest whole paths")
    for p, labels in sorted(paths.items(), key=lambda kv: -len(kv[1]))[:12]:
        print("  %-12s %4d digits  %s" % ("-".join(map(str, p)), len(labels),
                                          digits_of(torch.tensor(labels))))
    draw_walk(nodes, tops, test, ops, k)
    draw_vops(nodes, ops)
    draw_program(tops, x, y, base, test, ops, k)
    draw_digits(tops, x, y, test, ops, k)


def draw_walk(nodes, tops, test, ops, k):
    ticks = len(nodes)
    tall = max(len(r) for r in nodes)
    fig = plt.figure(figsize=(4.2 * ticks + 1, 1.25 * tall + 1.6), facecolor="white")
    back = fig.add_axes([0, 0, 1, 1])
    back.set_axis_off()
    back.set_xlim(0, 1)
    back.set_ylim(0, 1)
    size_x, size_y = 0.9 / (4.2 * ticks + 1), 0.9 / (1.25 * tall + 1.6)
    where = {}
    for t, row in enumerate(nodes):
        cx = (t + 0.28) / ticks
        for j, (op, share, img, text, _) in enumerate(row):
            cy = 0.86 - (j + 0.5) * 0.8 / tall
            where[(t, op)] = (cx, cy)
            ax = fig.add_axes([cx - size_x / 2, cy - size_y / 2, size_x, size_y])
            lim = max(abs(img.min().item()), abs(img.max().item()), 1e-6)
            ax.imshow(img.numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_xticks([])
            ax.set_yticks([])
            back.text(cx + size_x * 0.6, cy + size_y * 0.18, "vop %d   %.0f%% of digits" %
                      (op, 100 * share), fontsize=10, color=INK, va="center")
            back.text(cx + size_x * 0.6, cy - size_y * 0.2, text, fontsize=9, color=MUTED,
                      va="center")
        back.text(cx, 0.9, "tick %d" % (t + 1), fontsize=12, color=INK, ha="center",
                  weight="bold")
    n = len(tops)
    for t in range(ticks - 1):
        flow = Counter(zip(tops[:, t].tolist(), tops[:, t + 1].tolist()))
        for (a, b), c in flow.items():
            if (t, a) in where and (t + 1, b) in where and c / n >= 0.01:
                (x0, y0), (x1, y1) = where[(t, a)], where[(t + 1, b)]
                back.plot([x0 + 0.21, x1 - size_x * 0.56], [y0, y1], color="#2a78d6",
                          lw=0.4 + 14 * c / n, alpha=0.45, solid_capstyle="round")
    back.text(0.02, 0.965, "The walk on MNIST: %d vops, %d fetched a tick, test %.3f" %
              (ops, k, test), fontsize=14, color=INK, weight="bold")
    back.text(0.02, 0.025, "A box is a vop at a tick: the average digit sent to it less the "
              "average of all digits (red: more ink, blue: less), and which digits those are. "
              "Lines: where its digits go on the next tick;\nthickness is how many. The "
              "best-matching vop of each tick is followed; vops with under 1% of digits are "
              "left out.", fontsize=8.5, color=MUTED)
    fig.savefig("docs/img/walk-mnist.png", dpi=110)
    plt.close(fig)


def draw_vops(nodes, ops):
    rows = [(t, n) for t, row in enumerate(nodes) for n in row]
    fig, ax = plt.subplots(figsize=(8.2, 0.36 * len(rows) + 1.8), facecolor="white")
    table = torch.stack([n[4] for _, n in rows]).numpy()
    lim = abs(table).max()
    im = ax.imshow(table, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(10))
    ax.set_xlabel("digit", color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["tick %d  vop %d  (%.0f%%; %s)" % (t + 1, n[0], 100 * n[1], n[3])
                        for t, n in rows], fontsize=8.5, color=INK)
    ax.set_title("What each vop does to the answer\nmean change in the readout's score for "
                 "each digit when that vop alone is applied to the digits sent to it",
                 fontsize=10.5, color=INK, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig("docs/img/vops-mnist.png", dpi=110)
    plt.close(fig)


def draw_program(tops, x, y, base, test, ops, k):
    import math
    from matplotlib.patches import FancyArrowPatch
    n, ticks = tops.shape
    used = [o for o, c in Counter(tops.flatten().tolist()).most_common() if c / (n * ticks) >= 0.005]
    fig = plt.figure(figsize=(11, 11), facecolor="white")
    back = fig.add_axes([0, 0, 1, 1])
    back.set_axis_off()
    back.set_xlim(0, 1)
    back.set_ylim(0, 1)
    pos, box = {}, 0.085
    for j, op in enumerate(used):
        a = math.pi / 2 - 2 * math.pi * j / len(used)
        pos[op] = (0.5 + 0.33 * math.cos(a), 0.5 + 0.33 * math.sin(a))
    flow = Counter()
    for t in range(ticks - 1):
        flow.update(zip(tops[:, t].tolist(), tops[:, t + 1].tolist()))
    for (a, b), c in flow.items():
        if a not in pos or b not in pos or c / n < 0.01:
            continue
        (x0, y0), (x1, y1) = pos[a], pos[b]
        width = 0.6 + 16 * c / n
        if a == b:  # ran twice running: a loop on the outside of the ring
            dx, dy = x0 - 0.5, y0 - 0.5
            r = math.hypot(dx, dy)
            ox, oy = dx / r, dy / r
            p0 = (x0 + ox * 0.05 - oy * 0.03, y0 + oy * 0.05 + ox * 0.03)
            p1 = (x0 + ox * 0.05 + oy * 0.03, y0 + oy * 0.05 - ox * 0.03)
            arrow = FancyArrowPatch(p0, p1, connectionstyle="arc3,rad=-2.2", arrowstyle="-|>",
                                    mutation_scale=14, lw=width, color="#eb6834", alpha=0.6)
        else:
            arrow = FancyArrowPatch((x0, y0), (x1, y1), connectionstyle="arc3,rad=0.18",
                                    arrowstyle="-|>", mutation_scale=12, lw=width,
                                    color="#2a78d6", alpha=0.4, shrinkA=38, shrinkB=38)
        back.add_patch(arrow)
    for op in used:
        cx, cy = pos[op]
        m = (tops == op).any(dim=1)
        img = (x[m].mean(dim=0) - base).view(28, 28)
        ax = fig.add_axes([cx - box / 2, cy - box / 2, box, box])
        lim = max(abs(img.min().item()), abs(img.max().item()), 1e-6)
        ax.imshow(img.numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks([])
        ax.set_yticks([])
        when = [t + 1 for t in range(ticks) if (tops[:, t] == op).float().mean() >= 0.01]
        dx, dy = cx - 0.5, cy - 0.5
        r = math.hypot(dx, dy)
        tx, ty = cx + dx / r * 0.115, cy + dy / r * 0.105
        back.text(tx, ty, "vop %d\nticks %s\n%s" % (op, ",".join(map(str, when)),
                  digits_of(y[m], least=0.15)), fontsize=9, color=INK, ha="center", va="center")
    back.text(0.03, 0.965, "The program on MNIST: one box per vop, whatever the tick", fontsize=14,
              color=INK, weight="bold")
    back.text(0.03, 0.935, "%d vops, %d fetched a tick, %d ticks, test %.3f" % (ops, k, ticks, test),
              fontsize=10, color=MUTED)
    back.text(0.03, 0.03, "Blue arrow: after this vop, that one ran (thickness: how many digits). "
              "Orange loop: the vop ran twice running. Under each box: the ticks it runs on and "
              "the digits sent to it.\nThe best-matching vop of each tick is followed; a vop is "
              "not a layer, the same one can run early for one digit and late for another.",
              fontsize=8.5, color=MUTED)
    fig.savefig("docs/img/program-mnist.png", dpi=105)
    plt.close(fig)


def draw_digits(tops, x, y, test, ops, k, show=5):
    """A row per digit: its commonest whole paths, the share of that digit on each, and the
    average image of the examples that took it."""
    ticks = tops.shape[1]
    fig = plt.figure(figsize=(15, 13.5), facecolor="white")
    back = fig.add_axes([0, 0, 1, 1])
    back.set_axis_off()
    back.set_xlim(0, 1)
    back.set_ylim(0, 1)
    print("the paths each digit takes")
    for d in range(10):
        m = y == d
        n = int(m.sum())
        paths = Counter(map(tuple, tops[m].tolist()))
        cy = 0.9 - d * 0.086
        ax = fig.add_axes([0.035, cy - 0.034, 0.05, 0.068])
        ax.imshow(x[m].mean(dim=0).view(28, 28).numpy(), cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])
        common = paths.most_common(show)
        covered = sum(c for _, c in common) / n
        back.text(0.095, cy + 0.012, "digit %d" % d, fontsize=13, color=INK, weight="bold")
        back.text(0.095, cy - 0.014, "%d examples\n%d different paths" % (n, len(paths)),
                  fontsize=8.5, color=MUTED, va="top")
        line = "  digit %d (%d): %d paths;" % (d, n, len(paths))
        for j, (path, c) in enumerate(common):
            cx = 0.215 + j * 0.146
            on = m & (tops == torch.tensor(path)).all(dim=1)
            ax = fig.add_axes([cx, cy - 0.034, 0.05, 0.068])
            ax.imshow(x[on].mean(dim=0).view(28, 28).numpy(), cmap="gray_r")
            ax.set_xticks([])
            ax.set_yticks([])
            back.text(cx + 0.056, cy + 0.012, "%.0f%%" % (100 * c / n), fontsize=13,
                      color="#2a78d6" if j == 0 else INK, weight="bold")
            back.text(cx + 0.056, cy - 0.016, " > ".join(map(str, path)), fontsize=9.5,
                      color=INK)
            line += "  %s %.0f%%" % ("-".join(map(str, path)), 100 * c / n)
        back.text(0.215 + show * 0.146 - 0.004, cy, "rest\n%.0f%%" % (100 * (1 - covered)),
                  fontsize=9, color=MUTED, va="center")
        print(line)
    back.text(0.035, 0.972, "The paths each digit takes", fontsize=15, color=INK, weight="bold")
    back.text(0.035, 0.95, "%d vops, %d fetched a tick, %d ticks, test %.3f. All 10,000 test "
              "digits. A path is the best-matching vop of each tick, in order." %
              (ops, k, ticks, test), fontsize=10, color=MUTED)
    back.text(0.035, 0.022, "Left: the average of all test examples of the digit. Then its %d "
              "commonest paths: the share of that digit's examples that took the path, the vops "
              "in order, and the average of the examples that took it." % show, fontsize=9,
              color=MUTED)
    fig.savefig("docs/img/digits-mnist.png", dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    main()
