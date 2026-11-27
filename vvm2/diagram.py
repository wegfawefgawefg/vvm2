"""A picture of one tick of the machine in vvm2/core.py, with real numbers.

    python -m vvm2.diagram        writes docs/img/machine.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from .core import SIDE, SLOT, Machine, address_of

n = 12
rng = np.random.default_rng(0)
mem = np.zeros((n, SLOT), dtype=np.float32)
mem[0, 0] = address_of(3, n)
mem[4, :3] = [address_of(6, n), address_of(9, n), address_of(3, n)]
op = np.eye(SIDE, dtype=np.float32)
op[0, 0], op[1, 1] = -1, 0.5
mem[3] = op.ravel()
mem[6] = (rng.integers(-3, 4, (SIDE, SIDE)) / 10).ravel()
loaded = mem[6].copy()
(pc, load, store, nxt, result), = Machine(mem, cores=[0]).tick()

fig = plt.figure(figsize=(14, 8), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14), ax.set_ylim(0, 8), ax.axis("off")
mono = dict(family="monospace", color="black")
ax.text(0.15, 7.75, "MEMORY: one list of slots. Every slot is the same shape: 64 numbers, an 8 by 8 grid.",
        fontsize=12, weight="bold", **mono)
x0, w, h, top = 1.6, 3.3, 0.48, 6.9
says = {0: "pc .273", 3: "the op", 4: "load .545   store .818   next .273", 6: "(loaded this tick)",
        9: "(the result is stored here)"}
y = {}
for i in range(n):
    y[i] = top - i * (h + 0.06)
    ax.add_patch(Rectangle((x0, y[i]), w, h, fill=i in says, facecolor="#e6e6e6", edgecolor="black", lw=1))
    ax.text(x0 - 0.1, y[i] + h / 2, "slot %2d" % i, ha="right", va="center", fontsize=9, **mono)
    ax.text(0.15, y[i] + h / 2, "%.3f" % address_of(i, n), ha="left", va="center", fontsize=8, **mono)
    ax.text(x0 + 0.1, y[i] + h / 2, says.get(i, ""), ha="left", va="center", fontsize=8.5, **mono)
ax.text(0.15, top + h + 0.1, "address", fontsize=8, **mono)
ax.text(x0, y[0] + h + 0.08, "slot 0 is a core: its first number is its pc", fontsize=8.5, **mono)
ax.add_patch(Rectangle((x0 - 0.04, y[4] - 0.04), w + 0.08, 2 * h + 0.14, fill=False, edgecolor="black", lw=2.2))
ax.text(x0 + w + 1.0, y[3] + h * 0.55, "a vop: an op and", fontsize=8.5, **mono)
ax.text(x0 + w + 1.0, y[3] + h * 0.05, "its addresses", fontsize=8.5, **mono)
for name, frm, to, rad in (("pc", 0, 3, -0.6), ("load", 4, 6, -0.9), ("store", 4, 9, -0.6), ("next", 4, 3, 1.6)):
    ax.add_patch(FancyArrowPatch((x0 + w, y[frm] + h / 2), (x0 + w, y[to] + h * (0.8 if name == "next" else 0.5)),
                                 arrowstyle="-|>", connectionstyle="arc3,rad=%s" % rad, mutation_scale=14,
                                 lw=1.2, color="black", linestyle="--" if name == "next" else "-"))
ax.text(x0 + w + 0.5, (y[0] + y[3]) / 2 + h / 2, "pc", fontsize=10, weight="bold", **mono)
ax.text(x0 + w + 0.5, (y[4] + y[6]) / 2 + h / 2 - 0.25, "load", fontsize=10, weight="bold", **mono)
ax.text(x0 + w + 0.95, (y[4] + y[9]) / 2 + h / 2, "store", fontsize=10, weight="bold", **mono)

def grid(left, values, title):
    a = fig.add_axes([left, 0.5, 0.135, 0.135 * 14 / 8])
    a.imshow(values.reshape(SIDE, SIDE), cmap="gray", vmin=-1, vmax=1)
    a.set_xticks(np.arange(-0.5, SIDE)), a.set_yticks(np.arange(-0.5, SIDE))
    a.grid(color="black", lw=0.4), a.tick_params(length=0, labelbottom=False, labelleft=False)
    a.set_title(title, fontsize=9.5, **mono)

grid(0.47, mem[3], "the op: slot %d" % pc)
grid(0.645, loaded, "what was loaded: slot %d" % load)
grid(0.82, result, "the result")
fig.text(0.625, 0.61, "x", fontsize=20, ha="center", **mono)
fig.text(0.8, 0.61, "=", fontsize=20, ha="center", **mono)
ax.text(6.45, 6.75, "matmul: an 8 by 8 grid times an 8 by 8 grid gives an 8 by 8 grid",
        fontsize=10.5, weight="bold", **mono)
steps = ["ONE TICK (every core does this at once)",
         "1. the core's pc lands on a slot. That slot is the op.",
         "   The slot after it holds the vop's addresses: load, store, next.",
         "2. the load address lands on a slot. That slot is loaded.",
         "3. the op is matmulled into what was loaded.",
         "4. the store address lands on a slot. The result replaces what was there.",
         "5. the core's pc becomes the vop's next (here: back to the same vop).",
         "",
         "Nothing steps. Nothing walks. A vop's addresses sit in an ordinary slot,",
         "so another vop can load them, change them and store them back.",
         "An op and a piece of data are the same kind of thing: any slot can be either."]
for i, line in enumerate(steps):
    ax.text(6.45, 3.45 - i * 0.3, line, fontsize=10, weight="bold" if i == 0 else "normal", **mono)
fig.savefig("docs/img/machine.png", dpi=110)
print("docs/img/machine.png")
