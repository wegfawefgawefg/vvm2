# Plan

In order. Each phase has a gate; a phase that fails its gate is written up and the plan is
changed, not pushed through. Status is updated as things land.

## Phase 0: toys, in RAM, in PyTorch

Every result is beside an MLP, a GRU and a small transformer, matched twice: once on
parameter count, once on compute per step. Both numbers are in every table.

0a. *(queued)* Foundation: the machine of `docs/design.md`, the three baselines, the
    training harness with parameter and compute accounting, the bank-use measurements, and
    the first tasks (0b to 0d) running end to end.
0b. Stored associations: random key to value, N items. What it shows: capacity per entry
    and per unit of compute. Expected to be the machine's best case.
0c. Modular addition and parity. What it shows: retrieved edits can compute, not only store.
0d. Pointer chasing: follow a learned map k times, one hop a tick. What it shows: the walk.
0e. Bank collapse against the remedies in `docs/design.md`, cheapest first (noise in the
    pick, in the query, in the state; then normalisation; then re-seeding).
    Noise here means literal noise added to the vectors while the machine runs (the
    owner: "brains are full of noise"), at a strength that can be set for each place: the
    state, the query (the address), a value as it is read, a value as it is written, and
    which entries are picked. Measure for each, beyond collapse: does a learned walk
    survive it (is each entry good for a neighbourhood of states, not one point), does it
    generalise better, and how coarse can stored values be made afterwards (a bank trained
    under value noise should survive few-bit storage, which is more entries per gigabyte).
    Later: a noise strength learned per entry, as a reading of how many bits that entry
    needs and how much it matters when the bank is shrunk.
0f. Delayed copy and n-back, with and without the data memory. What it shows: the state's
    bandwidth and whether the run-time store removes the limit.
0g. Kuzushiji-MNIST, then a character-level language model on a small text.
0h. Does it learn to learn. The owner's hope for making program memory and data memory one
    store: an entry the machine writes while running is found and applied exactly as a
    learned entry is, so the machine can add an operation to itself from one example.
    Test: every episode has a fresh random map; the machine is shown some of its pairs,
    then asked about them. Nothing learned in advance can answer; only what it wrote this
    episode can. Three arms: no run-time store; a separate data memory; one store (run-time
    writes land in the program bank as new entries). Then: do the writes still help when
    they are kept across episodes (the first step toward learning continually), and is
    "growing the bank" the same act as a run-time write.
    How a write lands without trampling what was learned, each as its own arm: (1) blend
    and renormalise: the entry becomes norm(old + strength x new), so a write shifts an
    entry and never replaces it, and every vector's size ("brightness") stays bounded;
    (2) stiffness by use: an entry that has been read or trained many times takes writes
    weakly, a rarely used one takes them strongly; (3) fresh entries for what matches
    nothing well. Also: brightness as a signal of its own, an entry's size standing for how
    much it is trusted, fading when unused, so the dim ones are what a shrink removes.
0i. Credit past the backprop window, after 0f works. Five ways to leave something around,
    each tried alone on a task whose gap is longer than the window (delayed copy, n-back,
    a long pointer chase), between two fixed lines: truncated backprop at that window (the
    floor) and backprop through the whole sequence (the ceiling). (1) Credit along
    lookups: the gradient goes from a read straight to the write it found. (2) A fading
    trace kept with each entry. (3) A learned guess of the future's gradient at the
    window's edge. (4) A usefulness score per entry or stored item, bootstrapped. (5) Side
    objectives: predict the next input, reconstruct what was stored. Report what each
    costs in memory and compute per step beside what it buys.

Gate: a clear win at equal compute on something past 0b, and the bank is not locked in.
Using little of the bank is fine when the score is good: a small task should need a small
part of it. Locked in means: the score is poor while entries sit unused; or the same few
entries are picked whatever the input; or a harder or a new task does not draw on more of
the bank. So the measurements are: entries used against task size (it should grow with
the task and stop when the task is solved), how much the picked entries change with the
input, and whether a second task recruits unused entries or overwrites the first's.

## Phase 1: measure access before building storage

On the trained models of phase 0: how skewed entry use is, how large the working set is
over a window, the hit rate of a cache of a given size, and how well the next lookups can
be predicted from the current state. This decides the cache and prefetch design.

## Phase 2: the tiered store

A C++ engine: values in a file, a hot cache in RAM, layout sorted by co-use, batched
asynchronous reads; many workers walking one bank. Inference first, then training with
sparse writes, on a bank deliberately larger than RAM. GPU for the dense part where it
pays.

## Phase 3: grow, shrink, keep learning

Grow by splitting the busiest entries or seeding new ones near queries that matched
poorly; shrink by dropping the least used and merging near-duplicates. Test: the same
model at half and at double the bank. Then: learn a second task without losing the first;
one store with a slow and a fast write path.

## Phase 4: scale

A byte-level language model with a bank of tens of gigabytes on NVMe, against the best
dense model that fits the GPU.

## Todo, not scheduled

Nothing here yet. (Credit past the backprop window moved up to 0i; its references:
sparse attentive backtracking, Ke et al. 2018; temporal value transport, Hung et al. 2019;
eligibility traces and e-prop; SnAp, Menick et al. 2020; synthetic gradients, Jaderberg et
al. 2016.)
