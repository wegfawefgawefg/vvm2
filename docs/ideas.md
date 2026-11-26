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

## Added 2026-11-26, after the first measurements

36. **Registers that fit the hardware** (owner asked; a proposal, not built): the state as a
    few registers of 16 numbers each, which is one 512-bit vector register or one 64-byte
    cache line. A vop names two registers to read and one to write, as a real instruction
    does, and is a 16 by 32 matrix: small enough to stay in the first-level cache and to
    run as a handful of vector instructions. Registers a vop does not name are untouched,
    so "do nothing to the rest" is free, and vops that touch different registers can run
    at the same time.
37. **The size of a vop as a dial**: a rank-n vop reads n numbers from the state and writes
    them back along n directions. Rank 1 is a single neuron; full rank is a whole layer.
    Small vops and a big bank, or big vops and a small one, at the same compute: which is
    better, and does it depend on the task. (On MNIST a full vop needs no bank; rank-2 vops
    gain from one.)
38. **A vvm with rank-1 vops is a layer of neurons of which only the best-matching few are
    run.** An ordinary layer runs all of them. Worth stating as the plain-words link to
    ordinary nets.
39. **Why use is a plateau and a cliff, not Zipf** (`scripts/use.sh`): the vops in use have
    similar shares and the rest are dead. Does a varied, larger task give a long tail, and
    do noise or re-seeding bring dead vops in.
40. **Devices by memory map** (owner): the vvm acts and senses through places in its own
    address space that are wired to the world: a mouse, a keyboard, an eye or eyes, limbs.
    Some registers are overwritten by the world every tick (sensors) and some are read by
    the world every tick (actuators); the vvm has to discover what they do. Possibly also
    arbitrary sockets for other kinds of input and output. No redesign per kind of input.
41. **An eye it moves, for position and scale** (follows from 40): today an image is written
    into the state through one flat map, so nothing learned at one place carries to another.
    If the image is a device the vvm looks at through a small window it can move and zoom,
    position and scale are handled by moving the eye, not by a special layer per format.
42. **Flat vops as the normal vop** (owner's picture of a vop): one vector that reads the
    state down to one number, and one that says where that number goes. A full 64 by 64 vop
    is 64 of those bundled; the "small vops" in the findings are 2 bundled. With flat vops
    the bank is a pool of single neurons and many are fetched per tick.
43. **Halting** (owner asked; first version built, `scripts/halt.sh`): the answer and a
    "done" signal are read after every tick; "done" is trained to say whether the current
    answer is right. Care needed: an untrained "done" sits near one half, so the bar must
    be high or its start pushed low; and halting is not used during training, only read.
44. **Fetch by a bar, not a fixed count** (owner: always exactly 32 is "jank"): run every vop
    whose key matches the address above a bar, few or many as the moment needs, with a cap
    for cost. Halting then needs no extra signal: when nothing matches, the vvm has stopped.
    The boot set becomes whatever vops match an empty memory, not "the top 32".
45. **One RAM, with vops as runs of slots** (owner: the vops and the scratchpad are the same
    thing; "it becomes live to me when its writing to its own ops"): on the vector RAM a vop
    is 33 slots of 16 numbers (its matrix column by column, and its bias). Put the bank in
    the same RAM as scratch, let a vop's read and write slots fall anywhere, and a vop that
    writes into another vop's slots is the vvm editing its own program. To find out: does it
    learn not to trample itself when the RAM is big enough, or must program and data be
    kept apart. Needs each run to have its own copy of the part it can write.

## Added 2026-11-26, while designing version 6 (`vvm2/core.py`)

46. **One memory, with the pc and the addresses as ordinary slots** (owner: memory and state
    should not be separate things). *Built: version 6.*
47. **One write rule everywhere** (owner: an op should just write, and a quiet op writes
    quietly): a slot moves toward what is written by its loudness. *Built.*
48. **A vop sets the next pc and the next addresses as well as its store** (the "wide" vop):
    faster per unit of work on this CPU (`scripts/benchtick.sh`). *Built.*
49. **Several reads a tick.** *Built as a switch; four reads lift MNIST from 0.939 to 0.970.*
50. **A vop sees its own pointers,** so it can step them ("one to the right of where I am").
    Without it pointer arithmetic is impossible. *Built as a switch (`see_head`); being
    tested.*
51. **Addresses shaped like the device** (owner: addressing may need to follow the shape of
    the input): an address is [region | position], and each device lays out its own slots:
    row and column for an image, time for audio. *Built for images as a switch; alone it
    scored worse; being tested with sharp pointers and vops that see their pointers.*
52. **Learned addresses,** as the less biased alternative to 51: every slot's address starts
    random and is trained.
53. **Adding a device is giving it an unused region code.** Fixed-width addresses, the space
    divided from the start, as a real memory map is. No existing address moves.
54. **The world announces a new device in a known slot** (an interrupt and a device table);
    the program learns what to do about it.
55. **A driver is fixed and dumb** (owner: shims would eat compute and take the work away
    from the vvm): it only lays a device's numbers out as slots. It may have control slots
    the vvm writes, such as where to look and how far to zoom, and data slots the device
    fills with a small fixed window resampled smoothly, so the gradient can say "look a
    little left". Position and scale then come from the vvm's own action.
56. **Exploration without a clock** (owner: a timed curriculum is against learning without
    end): the amount of noise comes from the machine's own state, more where it is unsure
    or where nothing matches or a slot is unfamiliar, less where it is confident.
57. **Curiosity** (owner: "pull an icm in"; it is in the original vvm notes): a vop guesses
    what its next load will return; the surprise draws exploration and is a training signal
    with no labels. Guard against staring at pure noise: value surprise that is shrinking.
58. **A curriculum of data, not of settings** (owner): the machine's rules stay fixed for
    life; the world shows it easy things first.
59. **Fetch by a bar with a cap, and grow a vop when nothing matches** (owner: "nothing
    matches" should expand the mind, not stop it).
60. **Guard against sprawl:** every new idea must be a switch on the one core, a device
    outside the machine, or a training signal. Nothing new goes inside the tick unless it
    replaces something there.

