"""The machine on images: MNIST, Kuzushiji-MNIST, Fashion-MNIST.

  python -m vvm2.image [TASK STEPS]        default mnist 12000

The world cuts the image into 49 patches of 4 by 4 pixels, one a slot, and reads the answer
from the first 10 numbers of one output slot after the last tick. Four reads a tick, 8
ticks. Beside it: an MLP of the same number of weights.
"""
import sys

import torch
import torch.nn as nn

from .common import TASKS, mlp_matched, params, patches, score, train
from .core import Core, grid_pos, rand_pos


class ImageVM(nn.Module):
    def __init__(self, ticks=8, grid=False, **core):
        super().__init__()
        self.core = Core(**{"reads": 4, **core})
        self.ticks = ticks
        self.in_pos = grid_pos(7) if grid else rand_pos(49, 3)
        self.scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, x):
        outs = self.core(patches(x), self.ticks, self.in_pos, rand_pos(1, 4))
        return outs[-1][:, 0, :10] * self.scale


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    torch.set_num_threads(8)
    data = TASKS[task](0)
    print("%s, %d steps\n| model | weights | train | test |\n|---|---|---|---|" % (task, steps),
          flush=True)
    torch.manual_seed(0)
    vm = ImageVM()
    for name, model in (("vvm", vm), ("mlp", mlp_matched(784, 10, params(vm)))):
        fit, test, _ = train(model, data, steps, 128, 3e-3, 0, wd=0.05, shift=2)
        print("| %s | %d | %.3f | %.3f |" % (name, params(model), fit, test), flush=True)


if __name__ == "__main__":
    main()
