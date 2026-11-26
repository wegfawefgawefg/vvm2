# vvm2

A vector virtual machine: a neural net shaped like a computer, built so that most of its
weights can live on disk.

The basic operations of a computer are load, store and mutate. Here each is learned and
fractional:

- **Registers**: a few state vectors, the machine's working state.
- **Program bank**: a very large table of learned entries, each a key and a value. A step
  makes a query from the state, finds the few nearest keys, and adds their values to the
  state as a small edit. Only those few entries are read, so the cost of a step does not
  grow with the size of the bank.
- **Data memory**: the same kind of lookup into a store the machine writes while it runs.
- **ALU**: a small dense block applied every tick; the fixed instruction set.
- **Ticks**: the same bank on every tick, several ticks per input, so the lookups form a
  walk through a program.

Why: a large dense model cannot be cut up to run from disk, because every weight is used on
every step. A machine that reads a handful of entries per step can keep terabytes of
weights on an SSD behind a cache, grow or shrink its bank to fit the machine it is on, and
be walked by many workers at once.

Status: just started. `docs/plan.md` is the plan and its gates, `docs/design.md` the
machine, `docs/findings.md` what has been measured so far. Second attempt; the first
(`vvm`, 2026) never learned, for reasons written in `docs/design.md`.

```
scripts/setup.sh        make .venv with CPU torch
scripts/smoke.sh        every task and model, tiny, end to end
```
