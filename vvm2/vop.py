"""vvm with real vops: an op is a small feed-forward layer applied to the state.

  python -m vvm2.vop mnist --ops 16
  python -m vvm2.vop xor | sin | spirals

One tick:
  address   q = norm(Q . state); the k ops whose keys score highest are fetched
  execute   each fetched op i is a matrix W_i and a bias b_i: out_i = act(W_i . state + b_i)
  write     state = norm(sum over the k of softmax(score)_i * out_i)        (--apply ff)
            state = norm(state + that sum)                                  (--apply res)
The same bank on every tick, `ticks` ticks an input. The input is written into the state by
a linear map and the answer is read from the state by another; nothing else computes.

An op holds d * d + d numbers (or 2 * d * rank + d with --rank), so a bank of a few ops is
already a few layers' worth of weights. The key of an op is a separate small vector.
"""
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classic import MLP, TASKS, mlp_matched, params, score, train


class VopBank(nn.Module):
    def __init__(self, d, ops, k=2, d_key=16, rank=0):
        super().__init__()
        self.d, self.ops, self.k, self.d_key, self.rank = d, ops, min(k, ops), d_key, rank
        self.query = nn.Linear(d, d_key)
        self.keys = nn.Parameter(torch.randn(ops, d_key) / math.sqrt(d_key))
        if rank:
            self.u = nn.Parameter(torch.randn(ops, d, rank) / math.sqrt(rank))
            self.v = nn.Parameter(torch.randn(ops, rank, d) / math.sqrt(d))
        else:
            self.w = nn.Parameter(torch.randn(ops, d, d) / math.sqrt(d))
        self.b = nn.Parameter(torch.zeros(ops, d))
        self.register_buffer("picks", torch.zeros(ops))
        self.register_buffer("blocked", torch.zeros(ops, dtype=torch.bool))  # for knock-outs
        self.last = None

    def forward(self, state):
        """state [B, d] -> the mixed output of the k fetched ops, [B, d]."""
        q = F.layer_norm(self.query(state), (self.d_key,))
        scores = (q @ self.keys.T).masked_fill(self.blocked, float("-inf"))
        sc, idx = scores.topk(self.k, dim=-1)                      # [B, k]
        w = torch.softmax(sc, dim=-1)
        if self.rank:
            mid = torch.einsum("bkrj,bj->bkr", self.v[idx], state)
            out = torch.einsum("bkir,bkr->bki", self.u[idx], mid)
        else:
            out = torch.einsum("bkij,bj->bki", self.w[idx], state)
        out = F.gelu(out + self.b[idx])
        self.last = idx.detach()
        if not self.training:
            self.picks += torch.bincount(idx.flatten(), minlength=self.ops).float()
        return (w.unsqueeze(-1) * out).sum(dim=1)

    def madds(self):
        op = 2 * self.d * self.rank if self.rank else self.d * self.d
        return self.d * self.d_key + self.ops * self.d_key + self.k * (op + self.d)


class VopVVM(nn.Module):
    def __init__(self, in_features, outputs, d=64, ticks=3, apply="ff", **bank):
        super().__init__()
        self.in_features, self.d, self.ticks, self.apply = in_features, d, ticks, apply
        self.proj = nn.Linear(in_features, d)
        self.bank = VopBank(d, **bank)
        self.readout = nn.Linear(d, outputs)

    def forward(self, x, record=None):
        state = F.layer_norm(self.proj(x), (self.d,))
        for _ in range(self.ticks):
            out = self.bank(state)
            state = F.layer_norm(state + out if self.apply == "res" else out, (self.d,))
            if record is not None:
                record.append(self.bank.last)
        return self.readout(state)

    def madds_total(self):
        return (self.in_features * self.d + self.ticks * self.bank.madds()
                + self.d * self.readout.out_features)


def use(model, tx):
    """Ops ever picked on the test inputs, the busiest op's share, how often two inputs
    share an op on the first tick, and how many different ops one input visits."""
    b = model.bank
    b.picks.zero_()
    model.eval()
    rec = []
    with torch.no_grad():
        model(tx[:2000], record=rec)
    share = b.picks / b.picks.sum()
    first = rec[0]
    half = len(first) // 2
    same = (first[:half].unsqueeze(2) == first[half:2 * half].unsqueeze(1)).any(dim=2)
    walk = torch.cat(rec, dim=1)                                  # [n, ticks * k]
    distinct = torch.tensor([len(set(r.tolist())) for r in walk[:500]]).float().mean()
    return int((b.picks > 0).sum()), share.max().item(), same.float().mean().item(), \
        distinct.item()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    a = p.add_argument
    a("task", choices=list(TASKS))
    a("--seeds", type=int, default=3)
    a("--steps", type=int, default=2000)
    a("--batch", type=int, default=128)
    a("--lr", type=float, default=3e-3)
    a("--d", type=int, default=64)
    a("--ticks", type=int, default=3)
    a("--ops", type=int, default=16)
    a("--k", type=int, default=2)
    a("--rank", type=int, default=0, help=">0: each op is a rank-r matrix")
    a("--apply", default="ff", choices=["ff", "res"])
    a("--wd", type=float, default=0.0, help="weight decay")
    a("--shift", type=int, default=0, help="mnist: move training images up to this many pixels")
    a("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    kind = TASKS[args.task](0)[4]
    what = "accuracy" if kind == "class" else "mean squared error"
    print("%s: train / test %s, %d seeds, %d steps; vvm: d %d, %d ops, %d fetched, %d ticks, "
          "apply %s" % (args.task, what, args.seeds, args.steps, args.d, args.ops, args.k,
                        args.ticks, args.apply))
    print("| model | params | madds | train | test | ops used | busiest op | "
          "two inputs share an op | ops visited by one input | s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    fmt = "%.1e" if kind == "real" else "%.3f"
    for name in ["vvm", "mlp, same params", "mlp, same madds"]:
        rows = []
        for seed in range(args.seeds):
            data = TASKS[args.task](seed)
            ins = data[0].shape[1]
            outs = int(data[1].max()) + 1 if kind == "class" else 1
            torch.manual_seed(seed)
            ref = VopVVM(ins, outs, d=args.d, ticks=args.ticks, apply=args.apply,
                         ops=args.ops, k=args.k, rank=args.rank)
            if name == "vvm":
                model = ref
            elif name == "mlp, same params":
                model = mlp_matched(ins, outs, params(ref))
            else:
                best = min(range(2, 2049), key=lambda w: abs(
                    MLP(ins, outs, w).madds_total() - ref.madds_total()))
                model = MLP(ins, outs, best)
            tr, te, secs = train(model, data, args.steps, args.batch, args.lr, seed,
                                 wd=args.wd, shift=args.shift)
            u = use(model, data[2]) if name == "vvm" else None
            rows.append((params(model), model.madds_total(), tr, te, u, secs))
        mean = lambda j: sum(r[j] for r in rows) / len(rows)
        cols = ["-"] * 4
        if rows[0][4] is not None:
            m = [sum(r[4][j] for r in rows) / len(rows) for j in range(4)]
            cols = ["%.0f of %d" % (m[0], args.ops), "%.2f" % m[1], "%.2f" % m[2], "%.1f" % m[3]]
        print("| %s | %d | %d | %s | %s | %s | %.0f |" % (
            name, rows[0][0], rows[0][1], fmt % mean(2), fmt % mean(3), " | ".join(cols),
            mean(5)))


if __name__ == "__main__":
    main()
