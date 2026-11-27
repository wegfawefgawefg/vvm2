"""Program bank: product-key addressing, a sparse top-k read, and use counters.

N entries, N a square. Entry (a, b) has the key (subkey1[a], subkey2[b]) and a separate
value. A query is split in two halves; the score of entry (a, b) is half1 . subkey1[a] +
half2 . subkey2[b]. The exact top k of the N entries lie inside the top k of each half, so
a lookup scores 2 * sqrt(N) sub-keys and reads k values; the rest of the bank is untouched.

Heads are several queries into the one bank (shared sub-keys, shared values).

Sparse gradients: the values are an nn.Embedding(sparse=True), so the gradient of a step
is a sparse tensor holding only the rows that were read. torch.optim.SparseAdam updates
those rows and their moments only. The sub-key tables are small (2 * sqrt(N) rows), are
all scored on every lookup, and are trained densely with the rest of the model.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_like(x):
    u = torch.rand_like(x).clamp_(1e-9, 1.0 - 1e-9)
    return -torch.log(-torch.log(u))


class Bank(nn.Module):
    def __init__(self, in_dim, entries, d_value, heads=2, k=8, d_key=32, qnorm="layer",
                 pick_from=0, gumbel=0.0, qnoise=0.0, value_std=0.1, sparse=True):
        super().__init__()
        side = math.isqrt(entries)
        assert side * side == entries, "entries must be a square number"
        assert d_key % 2 == 0 and k <= side
        self.in_dim, self.entries, self.side, self.d_value = in_dim, entries, side, d_value
        self.heads, self.k, self.d_key, self.qnorm = heads, k, d_key, qnorm
        # Noise switches, all active in training only (see pick and forward).
        self.pick_from = pick_from  # >k: take the exact top pick_from, keep k at random
        self.gumbel = gumbel        # scale of Gumbel noise on the sub-key scores
        self.qnoise = qnoise        # std of Gaussian noise on the normalised query
        self.query = nn.Linear(in_dim, heads * d_key)
        self.subkeys = nn.Parameter(torch.randn(2, side, d_key // 2) / math.sqrt(d_key))
        self.values = nn.Embedding(entries, d_value, sparse=sparse)
        nn.init.normal_(self.values.weight, std=value_std)
        self.bn = nn.BatchNorm1d(heads * d_key) if qnorm == "batch" else None
        # Rows read since the last decay_read, kept only while track_read is on.
        self.track_read, self.read_rows = False, []
        # Use counters, per head. mass and hits are filled in eval mode while counting is
        # on; train_hits counts every pick made in training (the rows that got a gradient).
        self.counting = False
        self.register_buffer("mass", torch.zeros(heads, entries))
        self.register_buffer("hits", torch.zeros(heads, entries))
        self.register_buffer("train_hits", torch.zeros(heads, entries))

    def norm_query(self, q):
        if self.qnorm == "layer":
            return F.layer_norm(q, q.shape[-1:])
        if self.qnorm == "l2":
            return F.normalize(q, dim=-1) * math.sqrt(q.shape[-1])
        if self.qnorm == "batch":
            return self.bn(q.flatten(1)).view_as(q)
        return q

    def scores(self, state):
        """state [B, in_dim] -> sub-key scores of each half, [B, heads, side] each."""
        q = self.query(state).view(-1, self.heads, self.d_key)
        q = self.norm_query(q)
        if self.training and self.qnoise > 0:
            q = q + self.qnoise * torch.randn_like(q)
        half = self.d_key // 2
        return q[..., :half] @ self.subkeys[0].T, q[..., half:] @ self.subkeys[1].T

    def pick(self, s1, s2):
        """Entry ids [B, heads, k] and their clean scores (which carry the gradient)."""
        k, noisy = self.k, self.training
        c = max(k, self.pick_from) if noisy else k
        n1, n2 = s1, s2
        if noisy and self.gumbel > 0:
            n1 = s1 + self.gumbel * gumbel_like(s1)
            n2 = s2 + self.gumbel * gumbel_like(s2)
        m = min(c, self.side)
        v1, i1 = n1.topk(m, dim=-1)
        v2, i2 = n2.topk(m, dim=-1)
        pairs = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)  # [B, heads, m * m]
        c = min(c, m * m)
        flat = pairs.topk(c, dim=-1).indices  # the exact top c of all N entries
        if c > k:  # keep k of the c, uniformly at random
            keep = torch.rand(flat.shape, device=flat.device).topk(k, dim=-1).indices
            flat = flat.gather(-1, keep)
        a = i1.gather(-1, flat // m)
        b = i2.gather(-1, flat % m)
        return a * self.side + b, s1.gather(-1, a) + s2.gather(-1, b)

    def forward(self, state):
        """state [B, in_dim] -> edit [B, heads, d_value]."""
        s1, s2 = self.scores(state)
        idx, score = self.pick(s1, s2)
        w = torch.softmax(score, dim=-1)
        edit = (w.unsqueeze(-1) * self.values(idx)).sum(dim=-2)
        self.count(idx, w)
        if self.training and self.track_read:
            self.read_rows.append(idx.detach().flatten())
        return edit

    @torch.no_grad()
    def decay_read(self, rate):
        """Weight decay that stays sparse: shrink only the rows read since the last call.
        A row decays when it is read, not on every step."""
        if self.read_rows:
            rows = torch.cat(self.read_rows).unique()
            self.values.weight[rows] *= 1.0 - rate
        self.read_rows = []

    @torch.no_grad()
    def count(self, idx, w):
        i = idx.transpose(0, 1).reshape(self.heads, -1)
        if self.training:
            self.train_hits.scatter_add_(1, i, torch.ones_like(i, dtype=torch.float))
        elif self.counting:
            self.hits.scatter_add_(1, i, torch.ones_like(i, dtype=torch.float))
            self.mass.scatter_add_(1, i, w.transpose(0, 1).reshape(self.heads, -1))

    def reset_counts(self):
        self.mass.zero_()
        self.hits.zero_()

    def madds(self):
        """Multiply-adds of one lookup (all heads). The unread bank costs nothing."""
        h, k = self.heads, self.k
        return {
            "bank_query": self.in_dim * h * self.d_key + h * self.d_key,
            "bank_score": h * 2 * self.side * (self.d_key // 2),
            "bank_pick": h * k * k,
            "bank_read": h * k * self.d_value + h * k,
        }


def use_stats(mass, hits):
    """Per head: share of entries picked at all, entropy of the read weight over entries
    (as a share of log N, and as an effective number of entries), the top entry's share."""
    out = []
    for m, h in zip(mass, hits):
        p = m / m.sum().clamp_min(1e-12)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum().item()
        out.append({
            "used": (h > 0).float().mean().item(),
            "ent": ent / math.log(len(p)),
            "eff": math.exp(ent),
            "top": p.max().item(),
        })
    return out
