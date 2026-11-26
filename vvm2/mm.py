"""vvm with a memory map: the input and the output are their own domains, the state is a
scratchpad, and vops are flat.

  python -m vvm2.mm class [TASK]         classify: input domain the image, output domain 10 slots
  python -m vvm2.mm draw  [TASK]         generate: input domain a class and a style code,
                                         output domain a 784-pixel canvas -> docs/img/drawn-TASK.png

Memory, one row per example:
  input     read-only, set by the world (an image; or a class and a style code)
  scratch   the state: read and written, starts empty
  output    written by vops, read by the world (10 class slots; or a canvas)

One tick:
  address = norm(Q . norm(scratch))               only the scratchpad addresses the bank
  fetch the k vops whose keys match best          (k is 32, not 2: vops are small)
  each fetched vop:  x = read . [input, scratch, output] + bias      one number
                     write: "slot"    memory[its slot] += gain * tanh(x)
                            "vector"  memory += tanh(x) * its write vector
On the first tick the scratchpad is empty, so every example fetches the same vops: a boot
routine that looks at the input. There is no input map into the state and no readout map:
the answer is whatever the vops wrote into the output domain.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classic import TASKS, params, score, shifted, train


class MMVVM(nn.Module):
    def __init__(self, n_in, n_out, scratch=64, ops=1024, k=32, ticks=4, d_key=16,
                 write="slot"):
        super().__init__()
        self.n_in, self.n_out, self.scratch, self.ops = n_in, n_out, scratch, ops
        self.k, self.ticks, self.d_key, self.write = k, ticks, d_key, write
        can_write = scratch + n_out
        self.query = nn.Linear(scratch, d_key)
        self.keys = nn.Parameter(torch.randn(ops, d_key) / math.sqrt(d_key))
        self.read = nn.Parameter(torch.randn(ops, n_in + can_write) / math.sqrt(n_in + can_write))
        self.bias = nn.Parameter(torch.zeros(ops))
        if write == "slot":
            self.gain = nn.Parameter(torch.randn(ops) * 0.5)
            self.register_buffer("slot", torch.randperm(ops) % can_write)
        else:
            self.put = nn.Parameter(torch.randn(ops, can_write) / math.sqrt(can_write))
        self.scale = nn.Parameter(torch.tensor(3.0))
        self.register_buffer("picks", torch.zeros(ops))

    def forward(self, inp):
        mem = inp.new_zeros(len(inp), self.scratch + self.n_out)   # scratch, then output
        for _ in range(self.ticks):
            q = F.layer_norm(self.query(F.layer_norm(mem[:, :self.scratch], (self.scratch,))),
                             (self.d_key,))
            sc, idx = (q @ self.keys.T).topk(self.k, dim=-1)        # [B, k]
            w = torch.softmax(sc, dim=-1) * self.k
            seen = torch.cat([inp, mem], dim=1)
            x = torch.einsum("bkd,bd->bk", self.read[idx], seen) + self.bias[idx]
            act = torch.tanh(x) * w
            if self.write == "slot":
                mem = mem.scatter_add(1, self.slot[idx], act * self.gain[idx])
            else:
                mem = mem + torch.einsum("bk,bkd->bd", act, self.put[idx])
            if not self.training:
                self.picks += torch.bincount(idx.flatten(), minlength=self.ops).float()
        return mem[:, self.scratch:] * self.scale

    def madds_total(self):
        per = self.n_in + self.scratch + self.n_out
        put = 1 if self.write == "slot" else self.scratch + self.n_out
        return self.ticks * (self.scratch * self.d_key + self.ops * self.d_key +
                             self.k * (per + put))


def classify(task):
    torch.set_num_threads(8)
    data = TASKS[task](0)
    print("%s, memory-mapped vvm: scratch 64, 1,024 flat vops, 32 fetched a tick, 6,000 steps"
          % task)
    print("| write | ticks | params | madds | train | test | vops used |")
    print("|---|---|---|---|---|---|---|")
    for write, ticks in (("slot", 1), ("slot", 4), ("vector", 1), ("vector", 4)):
        torch.manual_seed(0)
        model = MMVVM(784, 10, ticks=ticks, write=write)
        fit, test, _ = train(model, data, 6000, 128, 3e-3, 0, wd=0.05, shift=2)
        model.picks.zero_()
        score(model, data[2], data[3], "class")
        print("| %s | %d | %d | %d | %.3f | %.3f | %d of %d |" % (
            write, ticks, params(model), model.madds_total(), fit, test,
            int((model.picks > 0).sum()), model.ops))


def draw(task):
    """Generate images. The input domain is a class (10 slots, one on) and a style code of
    8 numbers; the output domain is the canvas. Trained as an autoencoder: the style code of
    a training image is made by a small linear map from that image, and the vvm must paint
    the image back from its class and code. To generate, pick a class and draw a code from
    the spread of the training codes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.set_num_threads(8)
    x, y = TASKS[task](0)[:2]
    torch.manual_seed(0)
    model = MMVVM(18, 784, ticks=4, write="vector", ops=1024, k=32)
    coder = nn.Linear(784, 8)
    opt = torch.optim.AdamW(list(model.parameters()) + list(coder.parameters()), lr=2e-3,
                            weight_decay=0.01)
    steps = 8000
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        model.train()
        i = torch.randint(len(x), (128,), generator=g)
        code = coder(x[i])
        code = code + 0.1 * torch.randn_like(code)
        inp = torch.cat([F.one_hot(y[i], 10).float(), code], dim=1)
        loss = F.mse_loss(torch.sigmoid(model(inp)), x[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == steps - 1:
            print("  step %d: pixel error %.4f" % (step, loss.item()))
    model.eval()
    with torch.no_grad():
        codes = coder(x[:20000])
        fig, axes = plt.subplots(10, 12, figsize=(12, 10.4), facecolor="white")
        for d in range(10):
            mine = codes[y[:20000] == d]
            mean, spread = mine.mean(dim=0), mine.std(dim=0)
            real = x[:20000][y[:20000] == d][:2]
            new = mean + spread * torch.randn(10, 8)
            inp = torch.cat([F.one_hot(torch.full((10,), d), 10).float(), new], dim=1)
            made = torch.sigmoid(model(inp))
            for j in range(12):
                img = real[j] if j < 2 else made[j - 2]
                axes[d, j].imshow(img.view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
                axes[d, j].set_xticks([])
                axes[d, j].set_yticks([])
                for side in axes[d, j].spines.values():
                    side.set_color("#2a78d6" if j < 2 else "#cccccc")
        fig.suptitle("Drawn by the vvm (%s): each row a class. First two columns (blue): real "
                     "training images. The other ten: generated from the class and a random "
                     "style code." % task, fontsize=11, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig("docs/img/drawn-%s.png" % task, dpi=100)
    print("  wrote docs/img/drawn-%s.png" % task)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "class"
    which = sys.argv[2] if len(sys.argv) > 2 else "mnist"
    (classify if what == "class" else draw)(which)
