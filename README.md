# vvm2

A vector virtual machine: a neural net shaped like a computer, built so that most of its
weights can live on disk.

The basic operations of a computer are load, store and mutate. Here each is learned and
fractional:

- **State**: a vector, the vvm's working state and also its address into the bank.
- **Bank of vops**: each vop is a small feed-forward layer (a matrix and a bias) with a
  key. A tick makes an address from the state, fetches the few vops whose keys match
  best, and applies them to the state. The result is the next state.
- **Ticks**: the same bank on every tick, a few ticks per input, so the fetches form a walk
  through a program. On MNIST the walks form a tree: shared vops first, branches later.
- Nothing else computes. There is no dense layer on the state.

Why: a large dense model cannot be cut up to run from disk, because every weight is used on
every step. A vvm reads a handful of entries per step, so it can keep terabytes of
weights on an SSD behind a cache, grow or shrink its bank to fit the machine it is on, and
be walked by many workers at once.

Status: the core works on the ordinary tasks (xor, a sine curve, two spirals, MNIST),
level with an MLP of the same compute, with 4 to 16 vops. `docs/ideas.md` lists every idea
so far; `docs/plan.md` the order of attack. Second attempt; the first
(`vvm`, 2026) never learned, for reasons written in `docs/design.md`.

```
scripts/setup.sh        make .venv with CPU torch
scripts/smoke.sh        every task and model, tiny, end to end
scripts/classic.sh all  xor, a sine curve, two spirals, MNIST: the vvm against an MLP
scripts/tree.sh         do the walks form a tree
```
