"""Does what was learned in one place carry to another?

  python -m vvm2.shifttest [TASK]

Two questions, each for an MLP of the same size, version 6 with random addresses and version
6 with structured addresses (vvm2/core.py):
  a. trained on digits where they are, tested on digits moved one patch (4 pixels) in a
     random one of the eight directions: nothing in training asked for this
  b. trained on digits moved by up to one patch each way, tested on the same kind: how well
     can it learn to cope when it is shown that digits move
A move of exactly one patch leaves every patch's contents as they were, at the address next
door, so a program that steps its pointer could in principle read the moved digit unchanged.
"""
import sys

import torch

from .classic import TASKS, mlp_matched, params, score, train
from .core6 import Core


def moved(x, g, zero_ok=True):
    """Each image moved by -1, 0 or +1 patch (4 pixels) across and down, at random; with
    zero_ok False, never left where it was."""
    n = len(x)
    img = torch.nn.functional.pad(x.view(n, 28, 28), (4, 4, 4, 4))
    while True:
        oy = torch.randint(3, (n, 1, 1), generator=g) * 4
        ox = torch.randint(3, (n, 1, 1), generator=g) * 4
        if zero_ok:
            break
        still = (oy == 4) & (ox == 4)
        oy = torch.where(still, torch.zeros_like(oy), oy)
        break
    rows = oy + torch.arange(28).view(1, 28, 1)
    cols = ox + torch.arange(28).view(1, 1, 28)
    return img[torch.arange(n).view(n, 1, 1), rows, cols].reshape(n, 784)


def pointers(task):
    """The same test for the machine that can step its pointers: vops see the pc and the
    address slots (see_head), with random or structured addresses, soft or sharp pointers."""
    torch.set_num_threads(8)
    data = TASKS[task](0)
    x, y = data[2], data[3]
    g = torch.Generator().manual_seed(123)
    x_moved = moved(x, g, zero_ok=False)
    x_mixed = moved(x, g, zero_ok=True)
    print("%s, version 6 with vops that see their own pointers. Test accuracy." % task)
    print("| addresses | pointers | a: trained in place, in place | a: moved | b: trained "
          "moved, in place | b: moved or not |", flush=True)
    print("|---|---|---|---|---|---|", flush=True)
    for addresses, top in (("random", 0), ("grid", 0), ("grid", 3), ("grid", 1)):
        runs = []
        for augment in (None, moved):
            torch.manual_seed(0)
            m = Core(ticks=8, addresses=addresses, top=top, see_head=True)
            train(m, data, 6000, 128, 3e-3, 0, wd=0.05, augment=augment)
            runs.append(m)
        a, b = runs
        print("| %s | %s | %.3f | %.3f | %.3f | %.3f |" % (
            "structured" if addresses == "grid" else "random",
            "soft" if top == 0 else "nearest %d" % top,
            score(a, x, y, "class"), score(a, x_moved, y, "class"),
            score(b, x, y, "class"), score(b, x_mixed, y, "class")), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "pointers":
        return pointers(sys.argv[2] if len(sys.argv) > 2 else "mnist")
    task = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    torch.set_num_threads(8)
    data = TASKS[task](0)
    x, y = data[2], data[3]
    g = torch.Generator().manual_seed(123)
    x_moved = moved(x, g, zero_ok=False)       # every test digit moved one patch
    x_mixed = moved(x, g, zero_ok=True)        # moved or not, as in training b
    print("%s: test accuracy. 'in place': digits where they were. 'moved': every digit moved "
          "one patch." % task)
    print("| model | params | a: trained in place, in place | a: moved | b: trained moved, "
          "in place | b: moved or not |", flush=True)
    print("|---|---|---|---|---|---|", flush=True)

    def make(name):
        torch.manual_seed(0)
        if name == "mlp":
            return mlp_matched(784, 10, params(Core(ticks=8)))
        return Core(ticks=8, addresses="grid" if name.endswith("structured") else "random")

    for name in ("mlp", "v6, random addresses", "v6, structured"):
        a = make(name)
        train(a, data, 6000, 128, 3e-3, 0, wd=0.05)
        b = make(name)
        train(b, data, 6000, 128, 3e-3, 0, wd=0.05, augment=moved)
        print("| %s | %d | %.3f | %.3f | %.3f | %.3f |" % (
            name, params(a), score(a, x, y, "class"), score(a, x_moved, y, "class"),
            score(b, x, y, "class"), score(b, x_mixed, y, "class")), flush=True)


if __name__ == "__main__":
    main()
