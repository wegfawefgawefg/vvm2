# The machine, version 0

Plain statement of what is built first. Everything here is a starting point to be measured,
not a belief.

## Parts

- **State**: R registers of width d (start R = 1 to 4, d = 64 to 256). The input is written
  into the state on every tick it is present.
- **Program bank**: N entries. Entry i has a key and a separate value of width d.
  - Query: `q = norm(W_q . state)`, split in two halves.
  - Addressing by product keys: two tables of sqrt(N) sub-keys; the score of entry (a, b)
    is the first half's score with sub-key a plus the second half's with sub-key b. The
    top k of the N entries are found exactly from the top k of each half, with no index.
  - Read ("fractional load"): `edit = sum over the top k of softmax(score)_i * value_i`.
  - Apply: `state = state + edit`. Doing nothing is the default; an entry only edits.
  - Several heads may query at once.
- **Data memory**: a store of (key, value) pairs written at run time and read with the same
  top-k lookup. Write ("fractional store"): the machine emits a key, a value and a strength
  each tick. Version 0 keeps it apart from the program bank and clears it each episode.
  Whether the two can be one store with a slow write path (gradient) and a fast one (the
  machine's own writes) is an open question, and the answer to "will it learn continually".
- **ALU**: one small dense residual block (`state += W2 . act(W1 . norm(state))`), applied
  every tick. Its width is the machine's compute budget.
- **Ticks**: T ticks per input with the same weights. The readout is a linear map of the
  state after the last tick.

## Training

Backprop through the ticks and, for sequences, truncated backprop through time. The
gradient reaches the top-k values and, through the softmax over the k scores, the query and
the sub-keys. Entries outside the top k get nothing on that step. Sparse updates for the
bank, ordinary ones for the rest.

## Known failure to watch: bank collapse

A few entries win every lookup, the rest never get a gradient. Measured on every run: share
of entries used at all, entropy of use, the top entry's share. Remedies to try, cheapest
first: noise (in which entries are picked: sample from more than k, Gumbel top-k; in the
query; in the state), query normalisation, re-seeding entries that are never used near
queries that matched poorly.

## What the first vvm got wrong, and what this keeps

Measured on the original, run unchanged (thonksheet, docs/results-2026-11-25-vm-route.md):

1. Each op was as large as the state, so the state fell onto its nearest op within two
   ticks and the input was gone. Here: an entry is a small residual edit.
2. There was no way to do nothing. Here: the residual default.
3. An op was both its own address and its effect, so training it moved where it was found.
   Here: key and value are separate.
4. The gradient never reached which op was chosen. Here: the softmax over the retrieved few.
5. The input barely affected the walk. Here: the input is written into the state.

## Honest limits, stated before any run

- Entries read sparsely are worth much less per parameter than dense weights. They are
  cheap per unit of compute. A terabyte bank is not a terabyte dense model.
- The ALU's width sets how well the machine reasons; the bank adds knowledge and skills.
- One state vector is a bottleneck for sequences; the data memory is there to remove it.

## Prior work this leans on

- Product-key memory layers: Lample et al. 2019, "Large Memory Layers with Product Keys"
  (arXiv 1907.05242); Berges et al. 2024, "Memory Layers at Scale" (arXiv 2412.09764): the
  same lookup inside a transformer, up to 128 billion memory parameters, ahead of dense
  models with more than twice the compute, most on factual tasks.
- Sparse selection of what to compute: SLIDE (Chen et al., arXiv 1903.03129).
- Not in those: the same bank on every tick (a walk), a run-time store with the same
  addressing, the bank on disk behind a cache, growing and shrinking the bank.
