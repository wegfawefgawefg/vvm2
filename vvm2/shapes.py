"""Shapes: a generator with no download, and the tasks that write a canvas.

  python -m vvm2.image class shapes [STEPS]   name the shape (the classifier of vvm2/image.py)
  python -m vvm2.shapes draw [STEPS]          draw a named shape at a named size and place
  python -m vvm2.shapes gen TASK [STEPS]      generate fashion | mnist | kmnist images from a
                                              class and a few style numbers
  options after STEPS:  seed=0  ops=64  reads=2  ruler=1 (0: no ruler slots)

The generator draws a filled or outlined circle, square or triangle on a 28 by 28 canvas:
half-width r pixels, centre (cx, cy), edges one pixel soft. Naming trains on r from 3 to 7
with the centre in the left two thirds (cx up to 18), and is tested there, on large shapes
(r from 8 to 10) and on shapes whose centre is in the right third (cx from 19).

The canvas device, for drawing and generating. Output: 49 slots, the 4 by 4 patches of the
canvas row by row; a pixel is the sigmoid of 6 times its number. Input: the request, then a
ruler of 49 slots, one for each patch, holding that patch's row and column (two numbers and
two one-of-seven codes). The ruler is fixed numbers; the machine may walk it in step with
its write pointer to know where on the canvas it is. Input and output are lines here (the
core's two steps, next and previous). One tick a patch plus 4.

    draw   request slot: one-of-three shape, filled or not, r / 10, cx / 28, cy / 28
    gen    request slots: one-of-ten class; then 8 style numbers

What is and is not the machine in `gen`: the painter (request slots in, canvas out) is the
machine alone. The 8 style numbers of a training image are made by a small linear map from
that image (784 to 8), which is outside the machine and trained with it, so training is an
autoencoder with the class given. To generate, a class is named and the style numbers are
drawn from the spread of that class's training codes. Beside it: an MLP of the same number
of weights from the same request numbers to the 784 pixels.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import TASKS, mlp_matched, params
from .core import SLOT, Core

NAMES = ["circle", "square", "triangle"]


def draw_shapes(kind, filled, r, cx, cy):
    """[N] each -> [N, 784] images in 0..1."""
    ys, xs = torch.meshgrid(torch.arange(28.0), torch.arange(28.0), indexing="ij")
    dx, dy = xs - cx.view(-1, 1, 1), ys - cy.view(-1, 1, 1)
    rr = r.view(-1, 1, 1)
    far = torch.stack([(dx ** 2 + dy ** 2).sqrt() - rr,                     # circle
                       torch.maximum(dx.abs(), dy.abs()) - rr,              # square
                       torch.maximum(dy - rr, (dx.abs() - (dy + rr) / 2) * 0.894)])
    d = far[kind, torch.arange(len(kind))]              # distance outside the edge, pixels
    solid = (0.5 - d).clamp(0, 1)
    ring = solid - (-1.0 - d).clamp(0, 1)               # an outline 1.5 pixels thick
    return torch.where(filled.view(-1, 1, 1), solid, ring).flatten(1)


def sample(n, g, sizes=(3.0, 7.0), place="left"):
    """Random shapes. place: where the centre is, "left" (cx up to 18) or "right" (from 19)."""
    kind = torch.randint(3, (n,), generator=g)
    filled = torch.rand(n, generator=g) < 0.5
    r = sizes[0] + (sizes[1] - sizes[0]) * torch.rand(n, generator=g)
    cy = r + (27 - 2 * r) * torch.rand(n, generator=g)
    lo = r if place == "left" else r.clamp(min=19.0)
    hi = (27 - r).clamp(max=18.0) if place == "left" else 27 - r
    cx = lo + (hi - lo) * torch.rand(n, generator=g)
    return kind, filled, r, cx, cy


def shape_data(seed):
    """Naming: train x, y, test x, y, "class", and the two held-out tests."""
    g = torch.Generator().manual_seed(100 + seed)

    def make(n, **how):
        spec = sample(n, g, **how)
        return draw_shapes(*spec), spec[0]

    x, y = make(30000)
    tx, ty = make(4000)
    held = {"large sizes (r 8 to 10)": make(4000, sizes=(8.0, 10.0)),
            "centre in the right third": make(4000, place="right")}
    return x, y, tx, ty, "class", held


def ruler():
    """49 slots, one for each canvas patch: its row and column."""
    rows, cols = torch.arange(49) // 7, torch.arange(49) % 7
    return torch.cat([(rows / 6.0).unsqueeze(1), (cols / 6.0).unsqueeze(1),
                      F.one_hot(rows, 7).float(), F.one_hot(cols, 7).float()], dim=1)


def uncut(slots):
    """[B, 49, 16] patches -> [B, 784] pixels."""
    return slots.reshape(-1, 7, 7, 4, 4).permute(0, 1, 3, 2, 4).reshape(-1, 784)


class Painter(nn.Module):
    """The machine with the canvas device around it. request [B, n, 16] -> pixel logits."""

    def __init__(self, with_ruler=True, **core):
        super().__init__()
        self.core = Core(**{"reads": 2, "scratch": 8, "ops": 64, **core})
        self.register_buffer("ruler", ruler())
        self.with_ruler = with_ruler

    def forward(self, request, trace=False):
        world = request
        if self.with_ruler:
            world = torch.cat([request, self.ruler.expand(len(request), -1, -1)], dim=1)
        outs = self.core(world, 49 + 4, 49, trace=trace)
        return 6.0 * uncut(outs[-1])


def as_slots(flat):
    """[B, n] numbers -> [B, slots, 16], 16 to a slot, the last padded with zeros."""
    pad = (-flat.shape[1]) % SLOT
    return F.pad(flat, (0, pad)).view(len(flat), -1, SLOT)


def shape_request(kind, filled, r, cx, cy):
    return torch.cat([F.one_hot(kind, 3).float(), filled.float().unsqueeze(1),
                      (r / 10).unsqueeze(1), (cx / 28).unsqueeze(1), (cy / 28).unsqueeze(1)], 1)


class Flat(nn.Module):
    """A painter or an MLP, taking the request as a flat row of numbers."""

    def __init__(self, net, slots=None):
        super().__init__()
        self.net, self.slots = net, slots

    def forward(self, flat):
        if self.slots is None:
            return self.net(flat)
        return self.net(torch.cat([as_slots(flat[:, a:b]) for a, b in self.slots], dim=1))


def fit(model, batch_of, steps, seed, extra=(), size=64):
    """Train to paint: batch_of(g) gives the request numbers and the wanted pixels."""
    opt = torch.optim.AdamW(list(model.parameters()) + list(extra), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    for _ in range(steps):
        model.train()
        flat, want = batch_of(size, g)
        loss = F.mse_loss(torch.sigmoid(model(flat)), want)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()


def pixel_error(model, flat, want, hard=None):
    """Mean squared pixel error; `hard` sets the machine's whole steps for the run."""
    model.eval()
    if hard is not None:
        model.net.core.hard = hard
    with torch.no_grad():
        made = torch.cat([torch.sigmoid(model(flat[i:i + 500])) for i in range(0, len(flat), 500)])
    if hard is not None:
        model.net.core.hard = False
    return F.mse_loss(made, want).item(), made


def sheet(rows, title, path, labels=None):
    """A picture: each row a list of [784] images; `labels` name the columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n, m = len(rows), len(rows[0])
    fig, axes = plt.subplots(n, m, figsize=(1.05 * m, 1.05 * n + 0.6), facecolor="white",
                             squeeze=False)
    for i in range(n):
        for j in range(m):
            axes[i, j].imshow(rows[i][j].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for side in axes[i, j].spines.values():
                side.set_color("#cccccc")
            if labels and i == 0:
                axes[i, j].set_title(labels[j], fontsize=7, color="#333333")
    fig.suptitle(title, fontsize=9, x=0.01, ha="left", color="#111111")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / (1.05 * n + 0.6)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("  wrote %s" % path, flush=True)


def bank_line(painter, request):
    from .image import bank_use
    painter.core.hard = True
    with torch.no_grad():
        painter(request, trace=True)
    painter.core.hard = False
    vops = torch.stack([t[0] for t in painter.core.trace])
    return "vops used %.2f, entropy of use %.2f, top vop %.2f, pick changes with input %.2f" % (
        bank_use(vops, painter.core.ops))


def run_draw(steps, opt):
    """Task b: draw a named shape at a named size and place. Trained and tested on all
    sizes (r 3 to 10) and places; the test shapes are fresh draws."""
    seed = int(opt.get("seed", 0))
    core = dict(ops=int(opt.get("ops", 64)), reads=int(opt.get("reads", 2)))
    torch.manual_seed(seed)
    vm = Flat(Painter(opt.get("ruler", "1") == "1", **core), [(0, 7)])
    mlp = Flat(mlp_matched(7, 784, params(vm)))

    def batch_of(n, g, **how):
        spec = sample(n, g, sizes=(3.0, 10.0), **how)
        cx = spec[2] + (27 - 2 * spec[2]) * torch.rand(n, generator=g)     # any place
        spec = spec[:3] + (cx, spec[4])
        return shape_request(*spec), draw_shapes(*spec)

    test = batch_of(2000, torch.Generator().manual_seed(999))
    print("draw a shape, %d steps, seed %d, a bank of %d vops, %d read pointers, 53 ticks%s\n"
          "| model | weights | pixel error, fresh shapes |\n|---|---|---|" % (
              steps, seed, core["ops"], core["reads"],
              "" if vm.net.with_ruler else ", no ruler"), flush=True)
    made = {}
    for name, model in (("vvm", vm), ("mlp", mlp)):
        fit(model, batch_of, steps, seed)
        err, made[name] = pixel_error(model, *test, hard=False if name == "vvm" else None)
        print("| %s | %d | %.4f |" % (name, params(model), err), flush=True)
        if name == "vvm":
            sharp = vm.net.core.sharp()
            err, made["whole"] = pixel_error(model, *test, hard=True)
            print("| vvm, run with whole steps | %d | %.4f |" % (params(model), err), flush=True)
            print("  pointers exact as trained: %.2f; %s" % (
                sharp, bank_line(vm.net, as_slots(test[0][:500]))), flush=True)
    print("  for scale: a blank canvas has pixel error %.4f" % test[1].pow(2).mean())
    rows = [[test[1][i], made["vvm"][i], made["whole"][i], made["mlp"][i]] for i in range(8)]
    sheet(list(map(list, zip(*rows))), "Drawing a named shape. Rows: asked for; the vvm as "
          "trained; the vvm run with whole steps; an MLP of the same size.",
          "docs/img/shapes-drawn.png")


def run_gen(task, steps, opt):
    """Generate images of a data set from a class and 8 style numbers."""
    seed = int(opt.get("seed", 0))
    core = dict(ops=int(opt.get("ops", 64)), reads=int(opt.get("reads", 2)))
    x, y, tx, ty, _ = TASKS[task](seed)
    torch.manual_seed(seed)
    vm = Flat(Painter(opt.get("ruler", "1") == "1", **core), [(0, 10), (10, 18)])
    mlp = Flat(mlp_matched(18, 784, params(vm)))
    print("generate %s, %d steps, seed %d, a bank of %d vops, %d read pointers, 53 ticks\n"
          "| model | weights | pixel error, test images rebuilt from class and code |\n"
          "|---|---|---|" % (task, steps, seed, core["ops"], core["reads"]), flush=True)
    rows = [[x[y == d][0] for d in range(10)]]
    g2 = torch.Generator().manual_seed(seed + 5)
    for name, model in (("vvm", vm), ("mlp", mlp)):
        torch.manual_seed(seed)
        coder = nn.Linear(784, 8)          # outside the machine: makes the style numbers

        def batch_of(n, g):
            i = torch.randint(len(x), (n,), generator=g)
            code = coder(x[i])
            code = code + 0.1 * torch.randn(code.shape, generator=g)
            return torch.cat([F.one_hot(y[i], 10).float(), code], dim=1), x[i]

        fit(model, batch_of, steps, seed, extra=coder.parameters())
        with torch.no_grad():
            req = torch.cat([F.one_hot(ty[:2000], 10).float(), coder(tx[:2000])], dim=1)
            codes = coder(x[:20000])
        err, _ = pixel_error(model, req, tx[:2000], hard=False if name == "vvm" else None)
        print("| %s | %d (and %d in the map that makes the style numbers) | %.4f |" % (
            name, params(model), params(coder), err), flush=True)
        if name == "vvm":
            sharp = vm.net.core.sharp()
            err, _ = pixel_error(model, req, tx[:2000], hard=True)
            print("| vvm, run with whole steps | %d | %.4f |" % (params(model), err), flush=True)
            print("  pointers exact as trained: %.2f; %s" % (
                sharp, bank_line(vm.net, torch.cat(
                    [as_slots(req[:500, :10]), as_slots(req[:500, 10:])], dim=1))), flush=True)
        for hard in ((False, True) if name == "vvm" else (None,)):   # new images, 3 a class
            for _ in range(3 if hard is not True else 1):
                new = []
                for d in range(10):
                    mine = codes[y[:20000] == d]
                    code = mine.mean(0) + mine.std(0) * torch.randn(1, 8, generator=g2)
                    new.append(torch.cat([F.one_hot(torch.tensor([d]), 10).float(), code], 1))
                rows.append(list(pixel_error(model, torch.cat(new), tx[:10], hard=hard)[1]))
    sheet(rows, "Generated %s: a column a class. Row 1: a real image. Rows 2 to 4: the vvm "
          "from a class and random style numbers. Row 5: the vvm run with whole steps. "
          "Rows 6 to 8: an MLP of the same size." % task, "docs/img/generated-%s.png" % task)


if __name__ == "__main__":
    torch.set_num_threads(4)
    what = sys.argv[1] if len(sys.argv) > 1 else "draw"
    opt = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
    steps = next((int(a) for a in sys.argv[2:] if a.isdigit()), 2000)
    if what == "gen":
        run_gen(sys.argv[2], steps, opt)
    else:
        run_draw(steps, opt)
