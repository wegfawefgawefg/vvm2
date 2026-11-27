"""The vvm, version 0 (docs/design.md).

State: `regs` registers of width d. One tick, in order:
  1. input write: the input's embedding is added to register 0 (on the first tick of an
     input, or on every tick if hold is set);
  2. data memory read (if there is one): the edit is added to the last register;
  3. program bank lookup: head h adds its edit to register h mod regs;
  4. data memory write.
Applying an op means adding its vector to the state, weighted by how well its key matched.
There is no dense layer on the state: the bank of ops is the only thing that computes.
The same weights on every tick. The readout is a linear map of the normalised state after
the last tick.

feed "whole": the tokens of an example are one input (their embeddings summed), followed
by `ticks` ticks. feed "seq": each token is an input with `ticks` ticks of its own.

Parts switch off with use_bank / use_mem. Noise (training only): snoise in the
state here; the pick and query noise live in the bank.
"""
import torch
import torch.nn as nn

from .bank import Bank
from .memory import DataMemory


class VVM(nn.Module):
    def __init__(self, vocab, classes, d=64, regs=1, ticks=2, feed="whole", hold=False,
                 use_bank=True, use_mem=False, snoise=0.0,
                 mem_k=4, mem_key=32, **bank_args):
        super().__init__()
        self.d, self.regs, self.ticks, self.feed, self.hold = d, regs, ticks, feed, hold
        self.snoise, self.classes = snoise, classes
        dim = regs * d
        self.embed = nn.Embedding(vocab, d)
        self.bank = Bank(dim, d_value=d, **bank_args) if use_bank else None
        self.mem = DataMemory(dim, d, d_key=mem_key, k=mem_k) if use_mem else None
        self.out_norm = nn.LayerNorm(dim)
        self.readout = nn.Linear(dim, classes)
        if self.bank is not None:
            head_reg = torch.arange(self.bank.heads) % regs
            self.register_buffer("head_reg", head_reg)

    def inputs(self, x):
        """x [B, L] token ids -> list of inputs, each [B, d]."""
        e = self.embed(x)
        if self.feed == "whole":
            return [e.sum(dim=1)]
        return list(e.unbind(dim=1))

    def tick(self, state, inp):
        """state [B, regs, d]; inp [B, d] or None. Returns the next state."""
        if inp is not None:
            state = torch.cat([state[:, :1] + inp.unsqueeze(1), state[:, 1:]], dim=1)
        if self.training and self.snoise > 0:
            rms = state.detach().flatten(1).pow(2).mean(dim=1).sqrt().view(-1, 1, 1)
            state = state + self.snoise * rms * torch.randn_like(state)
        if self.mem is not None:
            got = self.mem.read(state.flatten(1))
            if got is not None:
                state = torch.cat([state[:, :-1], state[:, -1:] + got.unsqueeze(1)], dim=1)
        if self.bank is not None:
            state = state.index_add(1, self.head_reg, self.bank(state.flatten(1)))
        if self.mem is not None:
            self.mem.write(state.flatten(1))
        return state

    def read_out(self, state):
        return self.readout(self.out_norm(state.flatten(1)))

    def forward(self, x, record=None):
        """x [B, L] -> logits [B, classes]. record: a list that receives the state after
        every tick (for looking inside)."""
        state = torch.zeros(x.shape[0], self.regs, self.d, device=x.device)
        if self.mem is not None:
            self.mem.start()
        for inp in self.inputs(x):
            for t in range(self.ticks):
                state = self.tick(state, inp if (t == 0 or self.hold) else None)
                if record is not None:
                    record.append(state)
        return self.read_out(state)

    def madds(self, length):
        """Multiply-adds for one example, by part. Never counts the unread bank."""
        dim = self.regs * self.d
        n_ticks = self.ticks * (1 if self.feed == "whole" else length)
        parts = {"embed": length * self.d, "readout": dim * self.classes + dim}
        per_tick = {}
        if self.bank is not None:
            per_tick.update(self.bank.madds())
        for name, cost in per_tick.items():
            parts[name] = cost * n_ticks
        if self.mem is not None:
            parts.update(self.mem.madds(n_ticks))
        return parts
