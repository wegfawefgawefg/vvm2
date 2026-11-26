"""Looking inside the machine on pointer chasing: does the state walk f one hop a tick?

Two measurements, both on all (x, h) pairs of the task:

lens    No fitting. The machine's own readout is applied to the state after every tick.
        lens[h][j][i] is the share of the inputs with count h whose decoded item after
        tick j+1 equals f applied i times to x (i = 0 is x itself). A walk shows as a
        diagonal in the table for the largest h.

probe   One ridge-regression decoder, fitted on the states of all ticks for 70% of the
        start items, with the label "f applied min(tick, h) times to x" (one hop a tick,
        then hold). Scored per tick on the other 30% of start items. A decoder cannot get
        these right by storing start items; it can only if an item has the same code
        however the walk arrived at it.
"""
import torch
import torch.nn.functional as F


def walk_loss(model, x, meta):
    """Not a measurement: a training loss that asks for the walk outright. The readout of
    the state after tick j must give f applied min(j, h) times to x, for every tick."""
    record = []
    model(x, record=record)
    start, h = x[:, 0], x[:, 1] - meta["items"] + 1
    losses = [F.cross_entropy(model.read_out(s), meta["powers"][h.clamp(max=j), start])
              for j, s in enumerate(record, start=1)]
    return torch.stack(losses).mean()


@torch.no_grad()
def tick_states(model, x, batch=4096):
    """States after every tick, [ticks, n, regs * d]."""
    model.eval()
    chunks = []
    for i in range(0, len(x), batch):
        record = []
        model(x[i:i + batch], record=record)
        chunks.append(torch.stack([s.flatten(1) for s in record]))
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def walk_lens(model, states, start, h, powers):
    hops = powers.shape[0] - 1
    out = {}
    for count in range(1, hops + 1):
        sel = h == count
        rows = []
        for s in states:
            guess = model.read_out(s[sel].view(-1, model.regs, model.d)).argmax(dim=-1)
            rows.append([(guess == powers[i, start[sel]]).float().mean().item()
                         for i in range(hops + 1)])
        out[str(count)] = rows
    return out


@torch.no_grad()
def walk_probe(states, start, h, powers, seed=0, ridge=1e-2):
    ticks, n, dim = states.shape
    items = powers.shape[1]
    g = torch.Generator().manual_seed(seed)
    fit_item = torch.rand(items, generator=g) < 0.7
    fit = fit_item[start]
    tick_no = torch.arange(1, ticks + 1).unsqueeze(1).expand(ticks, n)
    label = powers[torch.minimum(tick_no, h.unsqueeze(0)), start.unsqueeze(0)]
    feats = (states - states.mean(dim=(0, 1))) / (states.std(dim=(0, 1)) + 1e-6)
    feats = torch.cat([feats, torch.ones(ticks, n, 1)], dim=-1)
    a = feats[:, fit].reshape(-1, dim + 1).double()
    onehot = F.one_hot(label[:, fit].reshape(-1), items).double()
    gram = a.T @ a + ridge * len(a) * torch.eye(dim + 1, dtype=torch.double)
    w = torch.linalg.solve(gram, a.T @ onehot).float()
    guess = (feats[:, ~fit] @ w).argmax(dim=-1)
    return (guess == label[:, ~fit]).float().mean(dim=1).tolist()


def look_inside(model, task):
    """The lens and the probe for a machine trained on the chase task."""
    meta = task["meta"]
    x = torch.cat([task["train_x"], task["test_x"]])
    start, h = x[:, 0], x[:, 1] - meta["items"] + 1
    states = tick_states(model, x)
    return {"lens": walk_lens(model, states, start, h, meta["powers"]),
            "probe": walk_probe(states, start, h, meta["powers"])}
