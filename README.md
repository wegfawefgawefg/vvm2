# vvm2

A vector virtual machine: a neural net shaped like a computer, built so that most of its
weights can live on disk.

The basic operations of a computer are load, store and mutate. Here each is learned and
fractional:

- **Memory**: one list of slots, 16 numbers each, every slot with an address. The pc and the
  read and write addresses are ordinary slots in it. The world writes some slots (input) and
  reads others (output).
- **Vops**: a bank of small learned ops, each a key and a matrix.
- **A tick**: fetch the vops nearest the pc; load the memory the addresses point at; one
  matmul gives a value to store and the next pc and addresses; store. One write rule for
  every store: a slot moves toward what is written by its loudness.
- Nothing else computes. There is no dense layer, no input map and no readout.

Why: a large dense model cannot be cut up to run from disk, because every weight is used on
every step. A vvm reads a handful of entries per step, so it can keep terabytes of
weights on an SSD behind a cache, grow or shrink its bank to fit the machine it is on, and
be walked by many workers at once.

Status: the machine (`vvm2/core.py`, `docs/design.md`) reaches 0.986 to 0.989 on MNIST
reading the image only through addresses it computes. It is being tested on tasks that need
steps: copying and reversing lists longer than it was trained on, and the lights game.
`docs/findings.md` has what has been measured, `docs/ideas.md` every idea, `docs/plan.md`
the order of attack. Five earlier forms of the machine are in `old/`. A first attempt
(`vvm`, 2026) never learned.

```
scripts/setup.sh                  make .venv with CPU torch
scripts/tape.sh copy|reverse|simon    lists and the lights game, tested longer than trained
scripts/image.sh mnist            the machine on images, beside an MLP
scripts/benchtick.sh              the cost of one tick on this CPU (C++)
old/scripts/                      the scripts of the earlier forms
```
