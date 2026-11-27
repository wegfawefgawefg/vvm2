"""Task generators. No downloads; everything is made from the seed.

Every task is a classification of a short tuple of tokens. A task is a dict:
  vocab, length, classes      sizes
  train_x [n, L], train_y [n] the training set (token ids are global: slot j's tokens have
                              their own id range unless the task says otherwise)
  test_x, test_y              the scoring set; for a storage task it is the training set
  test_group                  optional [n] int label per test example, for split accuracy
  shared_vocab                optional, True if the same token ids are used at every position
  chance                      accuracy of a constant guess
  meta                        whatever the look-inside code needs
"""
import torch


def assoc(seed, items=4096, digits=8, base=16, classes=64):
    """0b stored associations. `items` random pairs. A key is a string of `digits` digits
    in base `base`, drawn at random without repeats from the base**digits possible
    strings; digit j with value v is token j * base + v. The value is a class drawn
    uniformly from `classes`. There is nothing to generalise: the score is accuracy on
    the stored pairs themselves."""
    g = torch.Generator().manual_seed(seed)
    keys = torch.empty(0, digits, dtype=torch.long)
    while len(keys) < items:
        more = torch.randint(base, (2 * items, digits), generator=g)
        keys = torch.unique(torch.cat([keys, more]), dim=0)
    keys = keys[torch.randperm(len(keys), generator=g)[:items]]
    x = keys + torch.arange(digits) * base
    y = torch.randint(classes, (items,), generator=g)
    return dict(vocab=digits * base, length=digits, classes=classes, train_x=x, train_y=y,
                test_x=x, test_y=y, chance=1.0 / classes, meta={})


def split(x, y, train_share, g):
    order = torch.randperm(len(x), generator=g)
    n = int(round(train_share * len(x)))
    tr, te = order[:n], order[n:]
    return x[tr], y[tr], x[te], y[te]


def modadd(seed, p=97, train_share=0.6):
    """0c modular addition. Input (a, b), tokens a and p + b; target (a + b) mod p. All
    p * p pairs; a random train_share of them is trained on, the rest is the test."""
    g = torch.Generator().manual_seed(seed)
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    x, y = torch.stack([a, p + b], dim=1), (a + b) % p
    tx, ty, sx, sy = split(x, y, train_share, g)
    return dict(vocab=2 * p, length=2, classes=p, train_x=tx, train_y=ty, test_x=sx,
                test_y=sy, chance=1.0 / p, meta={})


def parity(seed, bits=12, train_share=0.6):
    """0c parity. Input `bits` bits, bit j with value v is token 2 * j + v; target the
    parity. All 2**bits strings; a random train_share is trained on, the rest is the
    test."""
    g = torch.Generator().manual_seed(seed)
    n = bits
    b = (torch.arange(2 ** n).unsqueeze(1) >> torch.arange(n)) & 1
    x, y = b + 2 * torch.arange(n), b.sum(dim=1) % 2
    tx, ty, sx, sy = split(x, y, train_share, g)
    return dict(vocab=2 * n, length=n, classes=2, train_x=tx, train_y=ty, test_x=sx,
                test_y=sy, chance=0.5, meta={})


def chase(seed, items=256, hops=4, holdout=0.3):
    """0d pointer chasing. f is a fixed random permutation of `items` items. Input
    (x, h): token x and token items + h - 1, h in 1..hops; target f applied h times to x.
    Training has every (x, 1), so all of f is shown, and a random 1 - holdout of the pairs
    with h >= 2. The test is the held-out pairs: they can be answered by walking f, and
    not by having stored the pair. test_group is h."""
    g = torch.Generator().manual_seed(seed)
    f = torch.randperm(items, generator=g)
    powers = [torch.arange(items)]
    for _ in range(hops):
        powers.append(f[powers[-1]])
    powers = torch.stack(powers)  # powers[j, x] = f applied j times to x
    xs = torch.arange(items).repeat(hops)
    hs = torch.arange(1, hops + 1).repeat_interleave(items)
    x, y = torch.stack([xs, items + hs - 1], dim=1), powers[hs, xs]
    held = (torch.rand(len(x), generator=g) < holdout) & (hs >= 2)
    return dict(vocab=items + hops, length=2, classes=items, train_x=x[~held],
                train_y=y[~held], test_x=x[held], test_y=y[held], test_group=hs[held],
                chance=1.0 / items, meta={"powers": powers, "hops": hops, "items": items})


def kvrecall(seed, pairs=4, keys=16, values=16, n_train=20000, n_test=2000):
    """A first exercise for the data memory (plan item 0f is the real one). A sequence of
    `pairs` (key, value) pairs with distinct keys, then one of the keys again; target that
    key's value. Key tokens 0..keys-1, value tokens keys..keys+values-1, shared across
    positions, so the order of the tokens matters. Train and test are separate samples."""
    g = torch.Generator().manual_seed(seed)
    n = n_train + n_test
    ks = torch.rand(n, keys, generator=g).argsort(dim=1)[:, :pairs]
    vs = torch.randint(values, (n, pairs), generator=g)
    which = torch.randint(pairs, (n, 1), generator=g)
    body = torch.stack([ks, keys + vs], dim=2).flatten(1)
    x = torch.cat([body, ks.gather(1, which)], dim=1)
    y = vs.gather(1, which).squeeze(1)
    return dict(vocab=keys + values, length=2 * pairs + 1, classes=values,
                train_x=x[:n_train], train_y=y[:n_train], test_x=x[n_train:],
                test_y=y[n_train:], chance=1.0 / values, shared_vocab=True, meta={})


TASKS = {"assoc": assoc, "modadd": modadd, "parity": parity, "chase": chase,
         "kvrecall": kvrecall}


def make_task(name, **settings):
    """Build a task from the settings it takes (seed among them); settings that are None
    or that the task does not have are dropped."""
    fn = TASKS[name]
    names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    task = fn(**{k: v for k, v in settings.items() if k in names and v is not None})
    task["name"] = name
    return task
