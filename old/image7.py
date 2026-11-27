"""Images through pointers that move on a grid and zoom.

  python -m old.image7 class TASK [STEPS]   classify mnist | kmnist | fashion | shapes
  python -m old.image7 look TASK            the look inside for the machine `class` saved:
                                            where the read pointers go, as text and a picture
                                            (which=mix, which=tuned: the machine saved by a
                                            run with mix= or tune=)
  python -m old.image7 time                 seconds a training step, nothing scored
  options after STEPS:  seed=0  ops=128  ticks=12  reads=4  scratch=8  mlp=1 (0 skips the MLP)
                        tune=0  further training steps run with whole steps (see `tune`)
                        mix=0   share of training passes run with whole steps (see `mixed`)

The image device. A 28 by 28 grey image is laid out as 66 input slots of 16 numbers, at
three levels:

    slot 0        the whole image shrunk to 4 by 4 (each number the mean of 7 by 7 pixels)
    slots 1..16   coarse: the image padded to 32 by 32 on the right and bottom, shrunk by
                  two, cut into a 4 by 4 grid of patches of 4 by 4; a patch covers 8 by 8
                  pixels, which is 2 by 2 fine patches (the last row and column are half
                  padding, because 28 is not a multiple of 8)
    slots 17..65  fine: the 49 patches of 4 by 4 pixels, a 7 by 7 grid

The device's six steps (`Core(moves=6)`): right, left, down, up within a level, stopping at
the edges; zoom in, to the patch below at the finer level (from a coarse patch, the upper
left of its four fine patches; from the whole image, the upper left of the four middle
coarse patches); zoom out, to the patch above. The whole-image slot is the first input slot.
The driver only lays numbers out; nothing here is learned except the machine and one number,
how sharply the answer is read. The answer is the first numbers of one output slot after the
last tick. Beside it: an MLP of the same number of weights on the 784 pixels.
"""
import os
import sys

import torch
import torch.nn as nn

from vvm2.common import TASKS, mlp_matched, params, score, train
from .core7 import ANCHORS, SLOT, Core

N_IN, COARSE0, FINE0 = 66, 1, 17
STEPS = ["stay", "right", "left", "down", "up", "zoom-in", "zoom-out", "to-in0", "to-inN",
         "to-sc0", "to-scN", "to-out0", "to-outN", "seek"]


def cut(img, n):
    """[B, 4n, 4n] -> [B, n*n, 16]: the 4 by 4 patches, row by row."""
    return img.reshape(-1, n, 4, n, 4).permute(0, 1, 3, 2, 4).reshape(-1, n * n, SLOT)


def levels(x):
    """[B, 784] -> [B, 66, 16]: the whole image, the coarse patches, the fine patches."""
    img = x.view(-1, 1, 28, 28)
    whole = nn.functional.avg_pool2d(img, 7).flatten(1).unsqueeze(1)
    half = nn.functional.avg_pool2d(nn.functional.pad(img, (0, 4, 0, 4)), 2)
    return torch.cat([whole, cut(half[:, 0], 4), cut(img[:, 0], 7)], dim=1)


def image_go():
    """The device's steps: go[m, i] is the input slot step m leads to from input slot i."""
    go = torch.arange(N_IN).repeat(6, 1)
    for first, n in ((COARSE0, 4), (FINE0, 7)):
        for r in range(n):
            for c in range(n):
                i = first + n * r + c
                go[0, i] = first + n * r + min(c + 1, n - 1)
                go[1, i] = first + n * r + max(c - 1, 0)
                go[2, i] = first + n * min(r + 1, n - 1) + c
                go[3, i] = first + n * max(r - 1, 0) + c
                if n == 4:
                    go[4, i] = FINE0 + 7 * (2 * r) + 2 * c
                    go[5, i] = 0
                else:
                    go[5, i] = COARSE0 + 4 * (r // 2) + c // 2
    go[4, 0] = COARSE0 + 4 * 1 + 1
    return go


def box(i):
    """The pixels input slot i covers: (level name, top, left, size), or None."""
    if i == 0:
        return "whole", 0, 0, 28
    if i < FINE0:
        return "coarse", 8 * ((i - 1) // 4), 8 * ((i - 1) % 4), 8
    if i < N_IN:
        return "fine", 4 * ((i - FINE0) // 7), 4 * ((i - FINE0) % 7), 4
    return None


class ImageVM(nn.Module):
    """The machine with the image device around it."""

    def __init__(self, outputs=10, ticks=12, **core):
        super().__init__()
        self.core = Core(**{"reads": 4, "scratch": 8, "moves": 6, **core})
        self.ticks, self.outputs = ticks, outputs
        self.register_buffer("go", image_go())
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.whole = False      # a training pass run with whole steps (see `mixed`)

    def forward(self, x, trace=False):
        if self.whole and self.training:      # the core in its running mode, gradients on
            self.core.eval()
            self.core.hard = True
        outs = self.core(levels(x), self.ticks, 1, trace=trace, go=self.go)
        if self.whole and self.training:
            self.core.train()
            self.core.hard = False
        return outs[-1][:, 0, :self.outputs] * self.scale

    def madds(self):
        """Multiply-adds for one image: the fetch, the matmul, and loading and seeking."""
        c = self.core
        slots = 1 + N_IN + c.scratch + 1
        fan_in = (c.reads + 2) * SLOT
        return self.ticks * (c.ops * SLOT + c.k * c.fan_out * fan_in
                             + 3 * (c.reads + 1) * slots * SLOT)

    def name(self, i):
        """A slot as text: W the whole image, C12 coarse row 1 column 2, F35 fine row 3
        column 5, s2 scratch, out the output slot, pc."""
        if i == 0:
            return "pc"
        i -= 1
        if i >= N_IN:
            return "s%d" % (i - N_IN) if i < N_IN + self.core.scratch else "out"
        b = box(i)
        return "W" if i == 0 else "%s%d%d" % (b[0][0].upper(), b[1] // b[3], b[2] // b[3])


def masses(vm, x):
    """Run x with a trace. Returns the best-matching vop at each tick [T, B]; where each
    pointer is when each tick loads, as the share of it on each memory slot [T, B, P, slots]
    (input slot i is memory slot i + 1); and the share of each step it then takes
    [T, B, P, steps]. Run with whole steps the shares are all 0 or 1."""
    core = vm.core
    with torch.no_grad():
        vm(x, trace=True)
    slots = 1 + N_IN + core.scratch + 1
    ends = [1, N_IN, 1 + N_IN, N_IN + core.scratch, slots - 1, slots - 1]
    start = torch.softmax(core.boot_at.detach(), dim=-1)      # the same sum the core makes
    if core.hard:
        start = nn.functional.one_hot(start.argmax(dim=-1), ANCHORS).float()
    at0 = start @ nn.functional.one_hot(torch.tensor(ends), slots).float()
    at = [at0.expand(len(x), -1, -1)] + [t[1] for t in core.trace]
    return (torch.stack([t[0] for t in core.trace]), torch.stack(at[:-1]),
            torch.stack([t[2] for t in core.trace]))


def paths(vm, x):
    """As `masses`, but the likeliest slot of each pointer [T, B, P] and the likeliest step
    [T, B, P]."""
    vops, at, step = masses(vm, x)
    return vops, at.argmax(dim=-1), step.argmax(dim=-1)


def bank_use(vops, ops):
    """Use of the bank over a set of runs, by the best-matching vop of each tick: share of
    vops ever used, entropy of use (1.0: all equally), the top vop's share, and how much the
    pick changes with the input (share of runs whose vop at a tick is not that tick's
    commonest, averaged over ticks)."""
    counts = torch.bincount(vops.flatten(), minlength=ops).float()
    p = counts / counts.sum()
    entropy = -(p[p > 0] * p[p > 0].log()).sum() / torch.log(torch.tensor(float(ops)))
    common = torch.stack([torch.bincount(t, minlength=ops).max() for t in vops]).float()
    return ((counts > 0).float().mean().item(), entropy.item(), p.max().item(),
            1 - (common / vops.shape[1]).mean().item())


def looked(vm, x):
    """How many different input slots the read pointers are on over a run, per image:
    all, and at the whole, coarse and fine levels. Means over x."""
    _, places, _ = paths(vm, x)
    at = places[:, :, :vm.core.reads] - 1                      # input slot numbers
    seen = torch.zeros(len(x), N_IN + vm.core.scratch + 1)
    at = torch.where(at < 0, torch.full_like(at, N_IN + vm.core.scratch), at)   # the pc: off
    seen.scatter_(1, at.permute(1, 0, 2).flatten(1), 1.0)
    seen = seen[:, :N_IN]
    return (seen.sum(1).mean().item(), seen[:, 0].mean().item(),
            seen[:, COARSE0:FINE0].sum(1).mean().item(), seen[:, FINE0:].sum(1).mean().item())


def tune(vm, data, steps, seed, shift=0):
    """Further training run with whole steps: every pointer takes its likeliest step, as it
    would when running, and only the values (what is stored, the next pc, what is sought)
    are taught; a whole step passes no gradient to the choice of step. The core is put in
    its running mode with gradients left on; nothing in the machine is changed."""
    from vvm2.common import shifted
    x, y = data[0], data[1]
    opt = torch.optim.AdamW(vm.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps)
    g = torch.Generator().manual_seed(seed + 1000)
    vm.eval()
    vm.core.hard = True
    for _ in range(steps):
        i = torch.randint(len(x), (128,), generator=g)
        xb = shifted(x[i], shift, g) if shift else x[i]
        loss = nn.functional.cross_entropy(vm(xb), y[i])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vm.parameters(), 1.0)
        opt.step()
        sched.step()
    vm.core.hard = False


def mixed(share, seed):
    """For common.train's each_step: run this share of training passes with whole steps
    (every pointer takes its likeliest step, as when running; such a pass teaches the
    values and gives the choice of step no gradient). The other passes are as usual."""
    g = torch.Generator().manual_seed(seed + 77)

    def each_step(model, done):
        model.whole = bool(torch.rand(1, generator=g) < share)
    return each_step


def rows(vm, data, label, fit=None):
    """The table rows for one trained machine: as trained, and run with whole steps."""
    x, y, tx, ty, kind = data
    out = []
    for hard in (False, True):
        vm.core.hard = hard
        test = score(vm, tx, ty, kind)
        sharp = vm.core.sharp()
        vops, _, _ = paths(vm, tx[:2000])
        used, ent, top, change = bank_use(vops, vm.core.ops)
        n, w, c, f = looked(vm, tx[:2000])
        how = "run with whole steps" if hard else "as trained"
        got = score(vm, x[:10000], y[:10000], kind) if hard or fit is None else fit
        out.append("| %s, %s | %d | %d | %.4f | %.4f | %.2f | %.1f (%.1f whole, %.1f coarse, "
                   "%.1f fine) | %.2f | %.2f | %.2f | %.2f |" % (
                       label, how, params(vm), vm.madds(), got, test, sharp, n, w, c, f, used,
                       ent, top, change))
    vm.core.hard = False
    return out


HEAD = ("| model | weights | multiply-adds an image | train | test | pointers exact | input "
        "slots looked at a run | vops used | entropy of use | top vop | pick changes with "
        "input |\n|---|---|---|---|---|---|---|---|---|---|---|")


def get_data(task, seed):
    if task == "shapes":
        from .shapes7 import shape_data
        return shape_data(seed)
    return TASKS[task](seed)


def saved(task, seed):
    return "out/image/%s-seed%d.pt" % (task, seed)


def classify(task, steps, opt):
    seed = int(opt.get("seed", 0))
    data = get_data(task, seed)
    tests = data[5] if len(data) > 5 else {}
    data = data[:5]
    outputs = int(data[1].max()) + 1
    shift = 2 if task in TASKS else 0
    core = dict(ops=int(opt.get("ops", 128)), reads=int(opt.get("reads", 4)),
                scratch=int(opt.get("scratch", 8)))
    torch.manual_seed(seed)
    vm = ImageVM(outputs, ticks=int(opt.get("ticks", 12)), **core)
    print("%s, %d steps, seed %d, %d ticks, %d read pointers, a bank of %d vops\n%s" % (
        task, steps, seed, vm.ticks, vm.core.reads, vm.core.ops, HEAD), flush=True)
    mix = float(opt.get("mix", 0))
    fit, _, secs = train(vm, data, steps, 128, 3e-3, seed, wd=0.05, shift=shift,
                         each_step=mixed(mix, seed) if mix else None)
    vm.whole = False
    name = "vvm, %.2f of passes with whole steps" % mix if mix else "vvm"
    for row in rows(vm, data, name, None if mix else fit):
        print(row, flush=True)
    os.makedirs("out/image", exist_ok=True)
    torch.save({"state": vm.state_dict(), "outputs": outputs, "ticks": vm.ticks, "core": core},
               saved(task + ("-mix" if mix else ""), seed))
    held_out(tests, "vvm", vm)
    if int(opt.get("tune", 0)):
        tune(vm, data, int(opt["tune"]), seed, shift)
        label = "vvm, then %s steps taught with whole steps" % opt["tune"]
        for row in rows(vm, data, label):
            print(row, flush=True)
        torch.save({"state": vm.state_dict(), "outputs": outputs, "ticks": vm.ticks,
                    "core": core}, saved(task + ("-mix" if mix else "") + "-tuned", seed))
        held_out(tests, label, vm)
    if opt.get("mlp", "1") == "1":
        mlp = mlp_matched(784, outputs, params(vm))
        mfit, mtest, msecs = train(mlp, data, steps, 128, 3e-3, seed, wd=0.05, shift=shift)
        print("| mlp | %d | %d | %.4f | %.4f | | | | | | |" % (
            params(mlp), mlp.madds_total(), mfit, mtest), flush=True)
        held_out(tests, "mlp", mlp)
    print("  seconds of training: vvm %.0f" % secs, flush=True)


def held_out(tests, label, model):
    """Scores on the held-out tests (shapes): sizes and places not trained on."""
    for name, (hx, hy) in tests.items():
        line = "  held out, %s: %s %.4f" % (name, label, score(model, hx, hy, "class"))
        if hasattr(model, "core"):
            model.core.hard = True
            line += "; run with whole steps %.4f" % score(model, hx, hy, "class")
            model.core.hard = False
        print(line, flush=True)


def load(task, seed):
    blob = torch.load(saved(task, seed))
    vm = ImageVM(blob["outputs"], ticks=blob["ticks"], **blob["core"])
    vm.load_state_dict(blob["state"])
    return vm.eval()


def timing(opt):
    """Seconds a training step at batch 128. Nothing is scored."""
    import time
    vm = ImageVM(ticks=int(opt.get("ticks", 12)), ops=int(opt.get("ops", 128)),
                 reads=int(opt.get("reads", 4)), scratch=int(opt.get("scratch", 8)))
    adam = torch.optim.AdamW(vm.parameters(), lr=1e-3)
    x, y = torch.rand(128, 784), torch.randint(10, (128,))
    start = time.time()
    for _ in range(20):
        loss = nn.functional.cross_entropy(vm(x), y)
        adam.zero_grad()
        loss.backward()
        adam.step()
    print("%d weights, %d multiply-adds an image, %.3f seconds a step" % (
        params(vm), vm.madds(), (time.time() - start) / 20))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("IMAGE_THREADS", 4)))
    what = sys.argv[1] if len(sys.argv) > 1 else "class"
    task = sys.argv[2] if len(sys.argv) > 2 else "mnist"
    opt = dict(a.split("=") for a in sys.argv[3:] if "=" in a)
    steps = next((int(a) for a in sys.argv[3:] if a.isdigit()), 3000)
    if what == "time":
        timing(dict(a.split("=") for a in sys.argv[2:] if "=" in a))
    elif what == "look":
        from .look7 import look
        look(task, int(opt.get("seed", 0)), opt.get("which", ""))
    else:
        classify(task, steps, opt)
