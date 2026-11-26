"""Data memory, version 0: a run-time store the machine writes while it runs.

One slot is appended per write (key, value, strength, all emitted from the state). A read
makes a query from the state, takes the top k slots by score, and returns the weighted sum
of their values, each scaled by its strength. The store is per episode: start() clears it.

Version 0 scans every slot on a read (exact, cost grows with the number of slots). That is
fine for episodes of tens of ticks; a long-running store needs the product-key addressing
of the bank and a rule for overwriting, neither of which is built here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DataMemory(nn.Module):
    def __init__(self, in_dim, d_value, d_key=32, k=4):
        super().__init__()
        self.in_dim, self.d_value, self.d_key, self.k = in_dim, d_value, d_key, k
        self.to_query = nn.Linear(in_dim, d_key)
        self.to_key = nn.Linear(in_dim, d_key)
        self.to_value = nn.Linear(in_dim, d_value)
        self.to_strength = nn.Linear(in_dim, 1)
        self.sharp = nn.Parameter(torch.tensor(8.0))  # keys and queries are unit vectors
        self.start()

    def start(self):
        self.keys, self.vals, self.strengths = [], [], []

    def write(self, state):
        """state [B, in_dim]: append one slot per batch row."""
        self.keys.append(F.normalize(self.to_key(state), dim=-1))
        self.vals.append(self.to_value(state))
        self.strengths.append(torch.sigmoid(self.to_strength(state)).squeeze(-1))

    def read(self, state):
        """state [B, in_dim] -> edit [B, d_value], or None while the store is empty."""
        if not self.keys:
            return None
        keys = torch.stack(self.keys, dim=1)            # [B, slots, d_key]
        vals = torch.stack(self.vals, dim=1)            # [B, slots, d_value]
        strengths = torch.stack(self.strengths, dim=1)  # [B, slots]
        q = F.normalize(self.to_query(state), dim=-1)
        score = self.sharp * torch.einsum("bd,bnd->bn", q, keys)
        top, i = score.topk(min(self.k, score.shape[1]), dim=-1)
        w = torch.softmax(top, dim=-1) * strengths.gather(1, i)
        picked = vals.gather(1, i.unsqueeze(-1).expand(-1, -1, self.d_value))
        return (w.unsqueeze(-1) * picked).sum(dim=1)

    def madds(self, n_ticks):
        """Multiply-adds over an episode of n_ticks ticks (one read and one write each).
        The read at tick t scans the t slots written so far."""
        scan = self.d_key * n_ticks * (n_ticks - 1) // 2
        read = n_ticks * (self.in_dim * self.d_key + self.k * self.d_value) + scan
        write = n_ticks * self.in_dim * (self.d_key + self.d_value + 1)
        return {"mem_read": read, "mem_write": write}
