"""Why does it help a vop to see the machine's own pointers?

  python -m vvm2.statecheck size      is it the information, or only a bigger vop?
  python -m vvm2.statecheck step      does it use them to step: do its reads follow a digit
                                      that has been moved?

size: MNIST, version 6, 8 ticks, 6,000 steps. A vop that sees its pc and addresses, against
a vop of the same size shown another run's pc and addresses (the same numbers of weights,
no information about itself), against the plain machine with a bank made larger until the
weight count is the same.

step: machines trained on digits moved by up to a patch. Each test digit is run where it is
and again moved one patch to the right. Compared tick by tick: is the same vop run; and
where is the read pointing: the centre of what it reads over the image, in patches. If the
machine steps its pointer with the digit, the centre moves one patch right and none down.
"""
import sys

import torch

from .classic import TASKS, params, score, train
from .core6 import Core
from .shifttest import moved


def size():
    torch.set_num_threads(8)
    data = TASKS["mnist"](0)
    print("mnist, version 6, 8 ticks, 6,000 steps: is it the information or the size?")
    print("| vop | vops | params | train | test |\n|---|---|---|---|---|", flush=True)
    for label, kw in [("plain", {}),
                      ("sees its own pc and addresses", {"see_head": True}),
                      ("sees another run's pc and addresses", {"see_head": "other"}),
                      ("plain, bank made larger", {"ops": 624})]:
        torch.manual_seed(0)
        m = Core(**{"ticks": 8, **kw})
        fit, test, _ = train(m, data, 6000, 128, 3e-3, 0, wd=0.05, shift=2)
        print("| %s | %d | %d | %.3f | %.3f |" % (label, m.ops, params(m), fit, test), flush=True)


def right_one_patch(x):
    img = torch.nn.functional.pad(x.view(-1, 28, 28), (4, 0, 0, 0))[:, :, :28]
    return img.reshape(-1, 784)


def centre(reads, model):
    """Read weights [n, slots] -> the row and column (in patches) of their centre over the
    image, and the share of the read that is on the image at all."""
    on = reads[:, model.head:model.head + model.n_in]
    side = 7
    rows = torch.arange(side).repeat_interleave(side).float()
    cols = torch.arange(side).repeat(side).float()
    mass = on.sum(dim=1).clamp_min(1e-9)
    return (on @ rows) / mass, (on @ cols) / mass, on.sum(dim=1)


def step():
    torch.set_num_threads(8)
    data = TASKS["mnist"](0)
    x, y = data[2][:2000], data[3][:2000]
    x2 = right_one_patch(x)
    print("mnist, version 6 trained on moved digits; each test digit where it is and one "
          "patch to the right")
    print("| addresses | sees its pointers | right, in place | right, moved | same vop on a "
          "tick | read centre moved right (patches) | moved down | share of reads on the "
          "image |\n|---|---|---|---|---|---|---|---|", flush=True)
    for addresses, see in (("random", False), ("random", True), ("grid", True)):
        torch.manual_seed(0)
        m = Core(ticks=8, addresses=addresses, see_head=see)
        train(m, data, 6000, 128, 3e-3, 0, wd=0.05, augment=moved)
        m.eval()
        with torch.no_grad():
            a = (m(x, trace=True).argmax(dim=1) == y).float().mean().item()
            t1 = m.trace
            b = (m(x2, trace=True).argmax(dim=1) == y).float().mean().item()
            t2 = m.trace
        same, dr, dc, on = [], [], [], []
        for (v1, r1, _), (v2, r2, _) in zip(t1, t2):
            same.append((v1 == v2).float().mean())
            ra, ca, ma = centre(r1[:, 0], m)
            rb, cb, mb = centre(r2[:, 0], m)
            keep = (ma > 0.5) & (mb > 0.5)
            on.append(keep.float().mean())
            if keep.any():
                dr.append((rb - ra)[keep].mean())
                dc.append((cb - ca)[keep].mean())
        mean = lambda v: (sum(v) / len(v)).item() if v else float("nan")
        print("| %s | %s | %.3f | %.3f | %.2f | %+.2f | %+.2f | %.2f |" % (
            "structured" if addresses == "grid" else "random", "yes" if see else "no",
            a, b, mean(same), mean(dc), mean(dr), mean(on)), flush=True)


if __name__ == "__main__":
    (size if (sys.argv[1:] or ["size"])[0] == "size" else step)()
