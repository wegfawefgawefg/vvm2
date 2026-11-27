"""A look inside a trained image classifier: where the read pointers go.

  python -m vvm2.image look TASK [seed=0] [tuned=1]

Loads the machine `python -m vvm2.image class TASK` saved, runs it with whole steps (every
pointer on one slot) and prints, for a few test images, a trace: at each tick the vop, the
slot each pointer loads from and the step it then takes. Then counts over 2,000 test images:
which levels are looked at, which steps are taken, how much the places change with the
image, and whether the pointers are on more ink than the same places hold on other images.
Draws the first few traces as docs/img/look-TASK.png.
"""
import os

import torch

from .image import FINE0, N_IN, STEPS, box, get_data, load, paths

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]       # one a read pointer


def trace_text(vm, x, y, n=3):
    vops, places, steps = paths(vm, x[:n])
    with torch.no_grad():
        said = vm(x[:n]).argmax(dim=1)
    reads = vm.core.reads
    for b in range(n):
        print("  image %d: class %d, the machine says %d" % (b, y[b], said[b]))
        print("    tick      " + " ".join("%8d" % t for t in range(len(vops))))
        print("    vop       " + " ".join("%8d" % v for v in vops[:, b]))
        for p in range(reads + 1):
            label = "write   " if p == reads else "read %d  " % (p + 1)
            print("    %s  " % label + " ".join("%8s" % vm.name(int(i)) for i in places[:, b, p]))
            print("      then    " + " ".join("%8s" % STEPS[int(s)] for s in steps[:, b, p]))


def ink_under(x, at):
    """Mean pixel value under each place. x [B, 784]; at [T, B, R] input slot numbers
    (anything off the image counts as no look). Returns the mean over looks on the image."""
    img = x.view(-1, 28, 28)
    total, count = 0.0, 0
    for i in at.unique().tolist():
        b = box(i) if 0 <= i < N_IN else None
        if b is None:
            continue
        _, top, left, size = b
        mean = img[:, top:top + size, left:left + size].mean(dim=(1, 2))      # [B]
        on = (at == i).float().sum(dim=(0, 2))                                 # looks, [B]
        total += (mean * on).sum().item()
        count += on.sum().item()
    return total / max(count, 1)


def counts(vm, x):
    vops, places, steps = paths(vm, x)
    reads = vm.core.reads
    at = places[:, :, :reads] - 1                      # input slot numbers; -1 is the pc
    level = torch.full_like(at, 3)                     # 0 whole, 1 coarse, 2 fine, 3 not image
    level[at == 0] = 0
    level[(at >= 1) & (at < FINE0)] = 1
    level[(at >= FINE0) & (at < N_IN)] = 2
    share = [(level == k).float().mean().item() for k in range(4)]
    print("  of all looks by read pointers: whole image %.2f, coarse %.2f, fine %.2f, not on "
          "the image (scratch, output, pc) %.2f" % tuple(share))
    for k, name in ((1, "coarse"), (2, "fine")):
        print("  share of runs with a look at the %s level: %.2f" % (
            name, (level == k).any(dim=2).any(dim=0).float().mean()))
    took = torch.bincount(steps[:, :, :reads].flatten(), minlength=len(STEPS)).float()
    took = took / took.sum()
    print("  steps taken by read pointers: " + ", ".join(
        "%s %.2f" % (STEPS[i], took[i]) for i in took.argsort(descending=True) if took[i] >= 0.005))
    common = torch.stack([torch.stack([torch.bincount(places[t, :, p]).max()
                                       for p in range(reads)]) for t in range(len(places))])
    print("  places that are not the commonest place for that pointer at that tick: %.2f "
          "(0: one path for every image)" % (1 - (common.float() / len(x)).mean()))
    paths_seen = len({tuple(places[:, b, :reads].flatten().tolist()) for b in range(len(x))})
    print("  different whole paths among %d images: %d" % (len(x), paths_seen))
    own, other = ink_under(x, at), ink_under(x.roll(1, 0), at)
    print("  ink under the read pointers (mean pixel): %.3f on the image itself, %.3f at the "
          "same places on another image, %.3f over a whole image" % (own, other, x.mean()))
    return places, steps


def picture(vm, x, y, path, n=4):
    """Rows: images. Columns: ticks. A box marks the patch each read pointer loads from."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    vops, places, steps = paths(vm, x[:n])
    with torch.no_grad():
        said = vm(x[:n]).argmax(dim=1)
    ticks, reads = len(vops), vm.core.reads
    fig, axes = plt.subplots(n, ticks, figsize=(1.25 * ticks, 1.55 * n + 0.9), facecolor="white",
                             squeeze=False)
    for b in range(n):
        for t in range(ticks):
            ax = axes[b, t]
            ax.imshow(x[b].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1, alpha=0.55,
                      extent=(0, 28, 28, 0))
            off = []
            for p in range(reads):
                i = int(places[t, b, p]) - 1
                spot = box(i) if 0 <= i < N_IN else None
                if spot is None:
                    off.append("%d:%s" % (p + 1, vm.name(i + 1)))
                    continue
                _, top, left, size = spot
                inset = 0.15 + 0.04 * size * p           # so boxes on one patch nest
                ax.add_patch(Rectangle((left + inset, top + inset), size - 2 * inset,
                                       size - 2 * inset, fill=False, lw=1.6,
                                       edgecolor=COLOURS[p % len(COLOURS)]))
                # the pointer's number at its own corner, so colour is not the only cue
                ax.text(left + (0.9 if p % 2 == 0 else size - 0.9),
                        top + (1.2 if p < 2 else size - 1.0), str(p + 1), fontsize=4.5,
                        color="#111111", ha="center", va="center")
            ax.set_xlim(-0.5, 28.5)
            ax.set_ylim(28.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color("#cccccc")
            ax.set_xlabel(" ".join(off), fontsize=6, color="#555555", labelpad=2)
            if b == 0:
                ax.set_title("tick %d" % t, fontsize=8, color="#333333")
            if t == 0:
                ax.set_ylabel("class %d, says %d" % (y[b], said[b]), fontsize=7, color="#333333")
    handles = [Rectangle((0, 0), 1, 1, fill=False, lw=1.6, edgecolor=COLOURS[p % len(COLOURS)],
                         label="read pointer %d" % (p + 1)) for p in range(reads)]
    fig.legend(handles=handles, loc="upper right", ncol=reads, fontsize=8, frameon=False)
    fig.suptitle("Where the read pointers are at each tick, run with whole steps. A box is the "
                 "patch a pointer loads from: the frame is the\nwhole-image slot, large boxes "
                 "the coarse level, small boxes the fine level. Under a panel: pointers that "
                 "are off the image (s scratch, out, pc).", fontsize=8, x=0.01, ha="left",
                 color="#111111")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (1.55 * n + 0.9)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("  wrote %s" % path)


def look(task, seed=0, tuned=False):
    vm = load(task + "-tuned" if tuned else task, seed)
    data = get_data(task, seed)
    tx, ty = data[2], data[3]
    print("look inside: %s, seed %d%s" % (task, seed, ", taught with whole steps" if tuned else ""))
    start = torch.softmax(vm.core.boot_at.detach(), dim=-1)
    ends = ["first input slot (the whole image)", "last input slot", "first scratch slot",
            "last scratch slot", "the output slot", "the output slot"]
    for p in range(vm.core.reads + 1):
        label = "write pointer" if p == vm.core.reads else "read pointer %d" % (p + 1)
        print("  %s starts at: %s (%.2f of its start is there)" % (
            label, ends[int(start[p].argmax())], start[p].max()))
    for hard in (False, True):
        vm.core.hard = hard
        print(" %s:" % ("run with whole steps" if hard else "as trained (likeliest slot of each "
                                                           "pointer)"))
        counts(vm, tx[:2000])
    print(" traces, run with whole steps (W the whole image, C12 coarse row 1 column 2, F35 "
          "fine row 3 column 5, s scratch, out the output slot):")
    trace_text(vm, tx, ty)
    picture(vm, tx, ty, "docs/img/look-%s%s.png" % (task, "-tuned" if tuned else ""))
    vm.core.hard = False
