"""The design as a picture: what is built, and the one-store version beside it.

  python -m old.diagram        -> docs/img/design.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, BLUE, ORANGE, GREEN = "#1a1a1a", "#555555", "#2a78d6", "#eb6834", "#1baf6b"


def box(ax, x, y, w, h, title, lines, edge=INK, fill="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006", fc=fill, ec=edge, lw=1.4))
    ax.text(x + 0.012, y + h - 0.03, title, fontsize=10.5, weight="bold", color=INK, va="top")
    ax.text(x + 0.012, y + h - 0.068, lines, fontsize=8.8, color=MUTED, va="top", linespacing=1.45)


def arrow(ax, a, b, text, color=INK, rad=0.0, dx=0.0, dy=0.012, ha="center"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.5, color=color,
                                 connectionstyle="arc3,rad=%g" % rad))
    ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, text, fontsize=8.6, color=color,
            ha=ha, va="bottom", linespacing=1.3)


def main():
    fig = plt.figure(figsize=(15, 8.6), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ---- left: what is built ----
    ax.text(0.03, 0.955, "What is built (vvm2/mm.py)", fontsize=14, weight="bold", color=INK)
    ax.text(0.03, 0.925, "memory: one copy per run, wiped between runs", fontsize=9.5, color=MUTED)
    box(ax, 0.03, 0.74, 0.16, 0.16, "input", "784 numbers (an image)\nset by the world\nvops may only read it", BLUE, "#eef5fd")
    box(ax, 0.03, 0.52, 0.16, 0.16, "scratch (the state)", "64 numbers\nstarts empty\nvops read and write it", INK)
    box(ax, 0.03, 0.30, 0.16, 0.16, "output", "10 numbers (the answer)\nvops write it\nthe world reads it", ORANGE, "#fdf1ea")
    ax.text(0.345, 0.925, "bank: the program. learned, shared by every run,\nnot changed while running",
            fontsize=9.5, color=MUTED, va="center")
    box(ax, 0.345, 0.50, 0.20, 0.37, "bank of 1,024 flat vops", "each vop is three vectors:\n\n  key      16 numbers\n           how it is found\n\n  read     858 numbers\n           what it looks at in memory\n\n  write    74 numbers\n           where its result goes", GREEN, "#eefaf3")
    arrow(ax, (0.19, 0.60), (0.345, 0.60), "1. the address is made\nfrom scratch only", INK, dy=0.012)
    arrow(ax, (0.445, 0.50), (0.445, 0.40), "2. fetch the 32 vops\nwhose keys match best", GREEN, dx=0.012, dy=-0.02, ha="left")
    box(ax, 0.345, 0.20, 0.20, 0.20, "the 32 fetched vops, each:", "x = read . [input, scratch, output]\n      (one number)\nmemory += tanh(x) * write", GREEN)
    arrow(ax, (0.19, 0.82), (0.345, 0.37), "", BLUE, rad=-0.22)
    ax.text(0.268, 0.80, "3. each vop reads\nall of memory", fontsize=8.6, color=BLUE, ha="center", va="bottom", linespacing=1.3)
    arrow(ax, (0.345, 0.27), (0.19, 0.55), "", INK, rad=-0.22)
    arrow(ax, (0.345, 0.24), (0.19, 0.38), "", ORANGE, rad=0.12)
    ax.text(0.268, 0.215, "4. and writes into\nscratch and output", fontsize=8.6, color=ORANGE, ha="center", va="top", linespacing=1.3)
    ax.text(0.03, 0.14, "That is one tick. Then again, 4 ticks in all, the same bank every tick.\n"
            "Tick 1: scratch is empty, so every input fetches the same 32 vops (a boot routine);\n"
            "after it scratch differs from input to input, and so do the fetches.\n"
            "There is no input map and no readout: the answer is what was written into output.",
            fontsize=9.2, color=INK, va="top", linespacing=1.5)

    ax.plot([0.585, 0.585], [0.04, 0.96], color="#cccccc", lw=1)

    # ---- right: the one-store version ----
    ax.text(0.615, 0.955, "The one-store version (not built)", fontsize=14, weight="bold", color=INK)
    ax.text(0.615, 0.925, "memory is one list of vectors; vops live in it and can write to it",
            fontsize=9.5, color=MUTED)
    rows = [("input vectors", "set by the world", BLUE, "#eef5fd"),
            ("output vectors", "read by the world", ORANGE, "#fdf1ea"),
            ("scratch vectors", "working memory", INK, "#ffffff"),
            ("vop", "key | read | write", GREEN, "#eefaf3"),
            ("vop", "key | read | write", GREEN, "#eefaf3"),
            ("vop", "key | read | write", GREEN, "#eefaf3"),
            ("...", "", MUTED, "#ffffff")]
    y = 0.86
    for name, what, edge, fill in rows:
        ax.add_patch(FancyBboxPatch((0.615, y - 0.058), 0.20, 0.052, boxstyle="round,pad=0.004",
                                    fc=fill, ec=edge, lw=1.3))
        ax.text(0.625, y - 0.032, name, fontsize=9.5, weight="bold", color=INK, va="center")
        ax.text(0.715, y - 0.032, what, fontsize=8.6, color=MUTED, va="center")
        y -= 0.072
    vop_y = 0.86 - 3 * 0.072 - 0.032          # the first vop row
    edge = 0.822
    arrow(ax, (edge, vop_y + 0.012), (edge, 0.86 - 0.032), "", BLUE, rad=0.55)
    arrow(ax, (edge, vop_y + 0.006), (edge, 0.86 - 2 * 0.072 - 0.032), "", INK, rad=0.9)
    ax.text(0.905, 0.80, "a vop reads vectors\nfrom the list", fontsize=8.8, color=BLUE, ha="left", va="center", linespacing=1.3)
    arrow(ax, (edge, vop_y), (edge, 0.86 - 0.072 - 0.032), "", ORANGE, rad=0.7)
    ax.text(0.905, 0.71, "and writes vectors\ninto it", fontsize=8.8, color=ORANGE, ha="left", va="center", linespacing=1.3)
    arrow(ax, (edge, vop_y - 0.012), (edge, 0.86 - 5 * 0.072 - 0.032), "", GREEN, rad=-0.7)
    ax.text(0.905, 0.55, "including onto\nother vops", fontsize=8.8, color=GREEN, ha="left", va="center", linespacing=1.3)
    ax.text(0.615, 0.33, "What changes:", fontsize=10, weight="bold", color=INK)
    ax.text(0.615, 0.30, "- A vop written while running is a new or changed instruction: the vvm\n"
            "  edits its own program. That is the 'store' that is missing today, and the\n"
            "  route to learning while running.\n"
            "- Scratch stops being a separate thing: it is just entries nobody has written\n"
            "  a vop into yet.\n\n"
            "What has to be decided before building it:\n"
            "- What is wiped between runs and what is kept. Today scratch is wiped and the\n"
            "  bank is kept; here they are the same list.\n"
            "- Many runs at once (a batch, or parallel workers) would all write the same\n"
            "  vops: each needs its own copy, or they share and collide.\n"
            "- How a write lands without trampling a learned vop.",
            fontsize=9.2, color=INK, va="top", linespacing=1.5)
    fig.savefig("docs/img/design.png", dpi=105)


if __name__ == "__main__":
    main()
