# Ideas

Everything proposed so far, listed and not done unless marked. One line each, who raised it,
and where it stands. Numbers are permanent; new ideas go at the end.

Where the core stands (2026-11-26): a vvm with matrix vops (`vvm2/vop.py`) learns xor, a
sine curve, two spirals and MNIST, level with an MLP of the same compute, with 4 to 16
vops, and its walks form a tree on MNIST (`scripts/classic.sh`, `scripts/tree.sh`).
Nothing below has been shown beyond that.

## What a vvm is

1. **A vop is a small feed-forward layer applied to the state** (owner). The state
   addresses the bank, the fetched vops are matmulled into the state, the result is the
   next state. No dense layer on the state anywhere. *Built.*
2. **Load, store and mutate, all fractional** (owner): fractional load, fractional store,
   smart addressing, a bank of learnable mutates. *Load and mutate built; store not.*
3. **Cycles** (owner): the same bank every tick, so the vvm should learn sequences and time
   series. *Ticks built; no sequence task run yet.*
4. **Walks should form trees** (owner): shared vops early, branching later; that is what
   makes compute and fetching cheap. *Seen on MNIST with 3 ticks; with 5 ticks the extra
   ticks do not branch further.*
5. **Richer ways to apply a vop**: low-rank vops (built as `--rank`, not measured), vops that
   gate or scale part of the state, vops that act on one register of several.
6. **Execute exactly one vop, or blend a few**: the original executed one sampled vop; this
   build blends the best two by how well their keys match, so the gradient reaches the
   choice. Which is better is open.
7. **The key of a vop apart from the vop** (built): in the original the op was its own
   address, so training it moved where it was found.
8. **Nothing counts ticks or decides when to stop.** A way for the vvm to halt, or to know
   which tick it is on.
9. **Supervising every tick** made a walk form on pointer chasing where supervising only
   the end did not (seen once, on the earlier build).

## Memory

10. **Program memory and data memory as one store** (owner), and the hope that follows:
    **it would learn to learn**. A vop or entry written while running is found and used
    exactly like a learned one, so the vvm can add to itself from one example.
11. **A run-time data memory** read and written with the same addressing. *A first version
    exists from the earlier build; not measured.*
12. **How a write lands without trampling what was learned**: blend and renormalise
    (owner: volume, magnitude, activity normalisation); stiffness by use (a much-used entry
    takes writes weakly); fresh entries for what matches nothing.
13. **Vector brightness** (owner): the size of a vector as a signal of its own, how much an
    entry is trusted, fading when unused, so the dim ones are what a shrink removes.
14. **Continual learning** (owner): hoped for, not a starting constraint. Test: a second
    task recruits unused vops and the first task is kept.

## The bank

15. **Not locked in** (owner): using little of the bank is fine when the score is good; the
    fault is the same vops whatever the input, or a new task that cannot reach unused ones.
16. **Noise** (owner): literal noise added to the vectors while running, at a strength set
    per place: the state, the address, a vop as it is read, a vop as it is written, which
    vops are picked. For exploration, for vops that cover a neighbourhood of states, for
    walks that survive jitter.
17. **Noise and cheap storage**: a bank trained under noise on its vops should survive
    few-bit storage, which is more vops per gigabyte.
18. **A noise strength learned per vop**: a reading of how many bits it needs and how much
    it matters when the bank is shrunk.
19. **Grow** (owner): add new vops, random or near existing ones, "exploring new ideas in
    vector space"; split the busiest; seed near states that matched nothing well.
20. **Shrink** (owner): fit a smaller computer by dropping the least used and merging
    near-duplicates. Test: the same vvm at half and at double the bank.
21. **Re-seeding dead vops**, as the fallback if noise is not enough against collapse.
22. **Addressing for very large banks**: product keys (the address search costs the square
    root of the bank's size, no index to rebuild) or a nearest-neighbour index.

## Learning

23. **Truncated backprop, as in ordinary practice** (owner): always intended.
24. **Credit past the backprop window, without a global pass** (owner asked; each to be
    tried alone): (a) credit along lookups, from a read straight to the write it found;
    (b) a fading trace kept with each vop; (c) a learned guess of the future's gradient at
    the window's edge; (d) a usefulness score per vop or stored item, bootstrapped;
    (e) side objectives: predict the next input, reconstruct what was stored.
25. **Rule-finding**: on modular addition with held-out pairs the earlier build memorised
    where dense nets found the rule. Not yet run with matrix vops.

## The hardware half

26. **The bank on disk** (owner): a vvm far larger than RAM, on an SSD "being sorted all
    the time", behind predictive fetch and cache layers. The reason for all of it: a huge
    dense model cannot be cut up to run from disk; this is designed to be.
27. **Many workers in one bank** (owner): one wide vvm on a GPU, or a thousand of them
    walking and training the same bank at once.
28. **Op registers** (owner, a metaphor for later): slots that keep fetched vops loaded, as
    a CPU's registers do, so a vop used tick after tick is not fetched again.
29. **Measure access before building storage**: how skewed vop use is, how big the working
    set is, the hit rate of a cache of a given size, how well the next fetches can be
    predicted from the state, how much of a walk is trunk.
30. **Layout sorted by co-use**, batched asynchronous reads, a hot cache in RAM; a C++
    engine for it once the Python version has settled what to build.

## Tasks and tests

31. **The ordinary tasks first** (owner): xor, sine, spirals, MNIST. *Done.* Next in line:
    Kuzushiji-MNIST, a small character-level language model.
32. **Sequence tasks**: delayed copy, n-back, pointer chasing, key-value recall.
33. **Learn-to-learn episodes**: a fresh random map every episode, shown some pairs, asked
    about them; only what was written this episode can answer.
34. **Always beside an MLP, a GRU and a small transformer**, matched on parameters and,
    separately, on compute; and the row with no bank at all (input write and readout only).
35. **Scale**: a byte-level language model with a bank of tens of gigabytes on NVMe against
    the best dense model that fits the GPU.
