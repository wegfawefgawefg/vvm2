"""Does the walk form a tree? Train a vvm on MNIST, then follow the best-matching op of each
tick: how many ops are in use at each tick, how many distinct paths there are so far, how
often two inputs are on the same op (same digit, different digit), and how well the path
alone names the digit.

  python -m vvm2.tree OPS FETCHED TICKS      e.g.  python -m vvm2.tree 16 2 3
"""
import sys, torch
from collections import Counter, defaultdict
from .classic import TASKS, train
from .vop import VopVVM
ops, k, ticks = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(8)
data = TASKS["mnist"](0)
torch.manual_seed(0)
m = VopVVM(784, 10, d=64, ticks=ticks, ops=ops, k=k)
tr, te, _ = train(m, data, 3000, 128, 3e-3, 0)
tx, ty = data[2][:4000], data[3][:4000]
m.eval(); rec = []
with torch.no_grad(): m(tx, record=rec)
top = torch.stack([r[:, 0] for r in rec], dim=1)        # [n, ticks] the best-matching op each tick
print("mnist, %d ops, %d fetched, %d ticks: test %.3f" % (ops, k, ticks, te))
h = len(tx) // 2
for t in range(ticks):
    a, b = top[:h, t], top[h:2*h, t]
    same_cls = ty[:h] == ty[h:2*h]
    eq = (a == b).float()
    pre = len(set(map(tuple, top[:, :t+1].tolist())))
    print("  tick %d: ops in use %2d | paths so far %3d | same top op: same digit %.2f, different digit %.2f"
          % (t + 1, len(set(top[:, t].tolist())), pre, eq[same_cls].mean(), eq[~same_cls].mean()))
paths = defaultdict(Counter)
for p, y in zip(map(tuple, top.tolist()), ty.tolist()): paths[p][y] += 1
right = sum(c.most_common(1)[0][1] for c in paths.values())
print("  full paths: %d; the path alone names the digit %.2f of the time (chance 0.11)" % (len(paths), right / len(tx)))
big = sorted(paths.items(), key=lambda kv: -sum(kv[1].values()))[:8]
for p, c in big:
    n = sum(c.values()); d, dn = c.most_common(1)[0]
    print("    path %s: %4d inputs, %.0f%% are digit %d" % ("-".join(map(str, p)), n, 100 * dn / n, d))
