# Findings

What has been measured, with the evidence. Grades: **solid** (two or more independent
results), **likely** (one clean result), **hint** (few seeds or one setting). All MNIST
numbers are test accuracy after 3,000 steps, 2 seeds, `scripts/classic.sh mnist ...`.

1. **The core learns the ordinary tasks.** [likely] A state, a bank of matrix vops fetched by
   key and applied to the state: xor, a sine curve, two spirals and MNIST, level with an
   MLP of the same compute (MNIST 0.978 against 0.978).
2. **With full-size vops, MNIST does not need the bank or the ticks.** [likely] One 64x64
   vop applied once: 0.976. Sixteen vops, two fetched: 0.979 at one tick, 0.979 at two,
   0.978 at three, 0.974 at six. So a tree of walks forms (`scripts/tree.sh`,
   `scripts/look.sh`), but the accuracy would be there without it: one vop applied each
   tick is a small recurrent net.
3. **An early leader becomes the trunk.** [likely: 3 seeds] The vop that is the best match
   for the most digits early in training (21% to 35% at the start) grows to 64% to 82% of
   first-tick picks. In one seed of three it is the very vop that led before any training.
4. **With small vops and compute held fixed, a bigger bank scores better.** [likely] Rank-2
   vops, two fetched, one tick (`--rank 2 --ticks 1`): a bank of 2 scores 0.918, of 16
   0.949, of 64 0.956, of 256 0.955, of 1,024 0.956, at the same compute per digit. This is
   the first sign of the idea itself: weights that sit unread cost nothing and still help.
   It levels off at about 64 vops, of which half are used; no trunk forms (the busiest vop
   is fetched for a fifth of digits). Each vop is readable: fetched for one or two digits,
   pushing the answer toward them (`scripts/lookvops.sh`).
5. **More ticks do not help yet, and with small vops they hurt.** [hint] Rank-2 vops, 256 in
   the bank: 0.955 at one tick, 0.915 at three. Nothing tells the vvm when to stop.
6. **A digit mostly takes one path, and the other paths are other handwriting.** [hint: one
   trained vvm] Share of a digit's examples on its commonest path: 91% for 0, 26% for 7;
   upright and slanted ones take different paths (`docs/img/digits-mnist.png`).
7. **The rarely fetched vops are doing real work.** [likely: two trainings] Small-vop bank
   of 64 (`scripts/lookvops.sh`): use is uneven but not a cliff (the busiest 4 take 32% to
   40% of fetches, the busiest 16 81% to 94%). Allowing only the busiest 2 vops: 0.32 to
   0.45; the busiest 4: 0.55 to 0.62; 8: 0.86 to 0.89; 16: 0.95; all: 0.956. Knocking out
   the busiest 4: 0.77 to 0.80.
8. **Ordinary practice matters and was missing.** [hint: one seed] 3,000 steps with no
   weight decay left what the vops read looking like noise. 12,000 steps with weight decay
   0.05 and training digits shifted up to 2 pixels: what each vop reads is a smooth stroke
   or curve (`docs/img/vops-small-trained.png` against `vops-small.png`). Weight decay
   alone lifts the test score to 0.963 (train 0.988); with the shifts it is 0.956 on both,
   so that vvm is then short of capacity, not overfitted.
9. **Use is a plateau and a cliff, not Zipf.** [hint: one seed each, `scripts/use.sh`] From
   the busiest down, the vops in use have similar shares (9% falling to about 2%) and then
   use drops to nothing: 33 of 64 in use, 55 of 256, 76 of 1,024. A bigger bank brings a
   few more vops in but most of it is never fetched. With 16 full vops and 3 ticks the
   busiest takes 23% of all fetches and 9 carry nearly everything.
10. **Shifted training digits, not weight decay, are what make a vop's reads smooth.**
    [hint: one seed] Weight decay alone leaves them speckled (`docs/img/vops-small-wd.png`);
    with shifts they are clean strokes (`vops-small-trained.png`).
11. **A small vop must add to the state, not replace it, or more ticks destroy it.** [likely:
    2 seeds] Rank-2 vops, 64 in the bank: replacing the state, 0.960 at one tick and 0.931
    at three; adding to it (`--apply res`), 0.967, 0.969 and 0.968 at one, three and six.
12. **With the answer asked for at every tick, the second and third ticks help a little and
    the rest do nothing.** [hint: one seed, `scripts/halt.sh`] Right if stopped at ticks 1
    to 6: MNIST 0.975 0.980 0.981 0.981 0.981 0.981; Kuzushiji-MNIST 0.889 0.905 0.909
    0.909 0.910 0.909. The walk does not spread over more vops with more ticks; it settles.
13. **Halting works as a mechanism.** [hint: one seed] "Done" above 0.99: MNIST 0.980 right
    at 2.3 ticks on average (43% of digits stop after one tick), Kuzushiji 0.909 at 3.9.
    Above 0.9: MNIST 0.976 at 1.07 ticks, Kuzushiji 0.898 at 1.5. The digits it chooses to
    run on are the hard ones (0.905 right against 0.979 for those stopped at once), and
    classes differ in the ticks they need (Kuzushiji: 1.3 to 2.1).
14. **First numbers off MNIST** [hint: one seed, no baseline run beside them yet]: 16 full
    vops, 3 ticks, weight decay, shifted images: MNIST 0.985, Kuzushiji-MNIST 0.906,
    Fashion-MNIST 0.874. Pictures of what each class's path uses: `docs/img/paths-*.png`.
15. **The memory-mapped vvm is the first form in which both the ticks and the bank clearly
    do work.** [hint: one seed, `scripts/mm.sh class`] Input and output as their own
    domains, the state a scratchpad that starts empty, 1,024 flat vops, 32 fetched a tick,
    no input map and no readout (the answer is what the vops wrote into 10 output slots).
    MNIST: writing one slot a vop, 0.927 at one tick and 0.976 at four, 703 of 1,024 vops in
    use; writing along a vector, 0.961 and 0.986, 756 in use. The first tick fetches the
    same 32 vops for every image (the scratchpad is empty): a boot routine.
16. **The same vvm draws.** [hint: one seed, `scripts/mm.sh draw`] Input domain a class and a
    style code of 8 numbers, output domain a 784-pixel canvas, 4 ticks: it paints
    recognisable, varied digits from a class and a random code (`docs/img/drawn-mnist.png`).
    Trained as an autoencoder, with a small linear map outside the vvm making the code from
    the training image; a separate vvm from the classifier, same code and same tick.
17. **Letting the input help choose the vops is far better at one tick and no better at
    four.** [hint: one seed] Memory-mapped vvm, writing along a vector. Address from the
    scratchpad only: MNIST 0.961 at one tick, 0.986 at four; Kuzushiji 0.789, 0.917.
    Address from the input too: MNIST 0.982, 0.983; Kuzushiji 0.922, 0.925. With the input
    in the address the first fetch already fits the image and later ticks add almost
    nothing; with the scratchpad only, the ticks do the work.
18. **In the painter the later ticks do real work.** [hint: one seed] The same painter with
    one tick (only the 32 fixed boot vops) ends at pixel error 0.0263 against 0.0163 with
    four ticks, and its digits are visibly worse (`docs/img/drawn-mnist-1tick.png` against
    `drawn-mnist.png`).
19. **A vvm on a vector RAM works, and there the ticks are everything.** [hint: one seed,
    `scripts/ram.sh`] Memory is slots of 16 numbers (49 image patches, 16 scratch, 1 output);
    a vop is an instruction that reads two fixed slots and adds a vector to a third; 2,048
    vops, 32 fetched a tick. MNIST: chance at one tick (the first tick only loads patches
    into scratch), 0.958 at two, 0.967 at four, 0.963 at eight; 1,131 of 2,048 vops in use
    at four ticks. Under the flat-vop version's 0.986, with vops that each see 32 numbers
    of memory, not all of it.
20. **On this CPU a wide vop does more work a second than a small one.** [hint: one
    unoptimised benchmark, one core of a Ryzen 9 9950X, floats, `scripts/benchtick.sh`] A
    small vop (16 by 32, one store a tick) against a wide one (64 by 32: the store, the next
    pc and both next addresses in one tick), fetched by a hash of the pc from a bank of a
    given size, every tick a jump to another vop:

    | bank | small vop, ns a tick | wide vop, ns a tick |
    |---|---|---|
    | 1 MB (in cache) | 66 | 113 |
    | 64 MB | 193 | 337 |
    | 2 GB | 286 | 529 |

    A wide tick costs 1.6 to 1.85 small ticks and does the work of up to four of them
    (a store and three pointer changes). The jump to the vop is most of the cost once the
    bank is out of cache: the same arithmetic takes four times longer at 2 GB than at 1 MB.
    Sixteen runs sharing the bank were no faster a tick, because the loop does not yet
    fetch ahead. The absolute numbers are a floor for effort, not a ceiling for speed.

## Version 6 (`vvm2/core.py`): one memory, pc and address slots, one write rule

All MNIST, 8 ticks, 256 vops, 4 blended a tick, one seed, 6,000 steps unless said.

21. **A machine that sees its input only through addresses it computes learns.** [likely]
    0.937 to 0.950 as first built; 0.970 after 24,000 steps with nothing else changed. It
    was undertrained, not broken.
22. **It was starved of input.** [likely] As built it reads 18 of 49 patches, each read
    spread over about 15 slots. Four reads a tick: 38 of 49 patches and 0.970 (from 0.939).
23. **A vop must see the machine's own pc and addresses, and it is the information that
    helps, not the size.** [likely: `python -m vvm2.statecheck size`] Plain 0.938; the same
    with a bank made larger to the same weight count 0.942; vops shown another run's pc and
    addresses 0.941; vops shown their own 0.952. It was always free to point a read at
    those slots and did not learn to.
24. **Asking for the answer at every tick, and more ticks, do nothing here.** [hint] 0.938
    against 0.939; with four reads 0.969 against 0.970; 16 ticks 0.965.
25. **Noise on the addresses hurts at every strength tried, on a timer or not.** [hint]
    None 0.937; fading from 0.3, 0.6, 1.0: 0.913, 0.903, 0.886; constant 0.3: 0.765. It
    reads fewer patches with noise, not more.
26. **Sharp pointers cannot be forced.** [hint] Structured addresses, trained on moved
    digits: soft pointers 0.874; reading only the 3 nearest slots 0.587; only the nearest
    one, chance. A hard cut gives the gradient nothing to follow.
27. **Moved digits** (every digit moved one patch; `python -m vvm2.shifttest`). [hint] Trained
    in place and tested moved, everything collapses (an MLP 0.277, version 6 0.18 to 0.19).
    Trained on moved digits: an MLP of the same size 0.980; version 6 0.851, 0.902 once its
    vops see their pointers; with structured addresses 0.767 and 0.874. Version 6 is well
    behind a plain net here.
28. **The first sign of a shared routine: with structured addresses and a view of its
    pointers, the same program runs on a moved digit and the reads begin to follow it.**
    [hint: one seed, `python -m vvm2.statecheck step`] Each test digit run in place and
    again one patch to the right, compared tick by tick:

    | addresses | sees its pointers | same vop on a tick | read centre moved right, patches | right, moved |
    |---|---|---|---|---|
    | random | no | 0.63 | 0.00 | 0.812 |
    | random | yes | 0.66 | +0.04 | 0.896 |
    | structured | yes | 0.97 | +0.24 | 0.845 |

    With random addresses the machine reads the same places wherever the digit is and runs
    different vops to cope. With structured addresses it runs the same vops (97%) and its
    reads shift a quarter of the way with the digit, in the right direction and not down.
    A quarter is not following, and that machine scores lower than the random one.
29. **Version 6 reaches the earlier versions' score once it reads more, sees its pointers and
    is trained longer.** [hint: one seed, `python -m vvm2.core best`, 12,000 steps] Four
    reads a tick and vops that see their pointers: 0.986. Eight reads: 0.989, reading 46 of
    49 patches. Sixteen ticks instead of eight: 0.986, no gain. The earlier best was 0.986
    (the flat-vop form). The cost is size on paper: 5.1 and 13.7 million weights, because
    each vop's matrix grew with what it sees; but only 24 to 30 of the 256 vops ever run,
    so about a tenth of those weights are in use.

## Tasks that need steps (`vvm2/tape.py`, the generalised core)

30. **It copies, reverses and plays the lights game perfectly at the lengths it was trained
    on, and does not carry any of them to longer ones.** [hint: one seed, 3,000 steps,
    `scripts/tape.sh copy|reverse|simon`] Share of items right / share of whole sequences
    right; lists trained on lengths 2 to 8, the lights game on 1 to 6.

    | task | model | trained length | longer | longer still | longest |
    |---|---|---|---|---|---|
    | copy (8, 12, 16, 24) | vvm | 1.00 / 1.00 | 0.70 / 0.00 | 0.55 / 0.00 | 0.39 / 0.00 |
    | | GRU | 1.00 / 1.00 | 0.57 / 0.00 | 0.50 / 0.00 | 0.35 / 0.00 |
    | reverse (8, 12, 16, 24) | vvm | 1.00 / 1.00 | 0.13 / 0.00 | 0.13 / 0.00 | 0.13 / 0.00 |
    | | GRU | 0.99 / 0.92 | 0.67 / 0.01 | 0.52 / 0.00 | 0.38 / 0.00 |
    | lights game (6, 8, 10, 12) | vvm | 1.00 / 1.00 | 0.45 / 0.00 | 0.35 / 0.00 | 0.29 / 0.00 |
    | | GRU | 1.00 / 1.00 | 0.80 / 0.08 | 0.67 / 0.00 | 0.61 / 0.00 |

    The machine stores a sequence that arrives one light at a time and plays it back, with
    nothing but its own stores to hold it. But it has learned the places, not the routine:
    on a longer copy it gets the first eight items and little after; on a longer reverse it
    is at chance (0.125), where the GRU still gets the most recent items. Not yet looked
    at: whether the pc cycles or runs straight through, and whether the pointers step by a
    fixed turn. Not tried: lists placed at random starts along the line, and a bank too
    small to hold an unrolled program, either of which should force a loop.
31. **Lists that start anywhere make it learn the routine, and the routine carries to longer
    lists.** [hint: one seed, 3,000 steps, `scripts/tape.sh copy 3000 shift=16`] Copy,
    trained on lengths 2 to 8, items right / whole sequences right at lengths 8, 12, 16, 24:

    | training | bank | 8 | 12 | 16 | 24 |
    |---|---|---|---|---|---|
    | lists always at the start of the line | 128 vops | 1.00 / 1.00 | 0.70 / 0.00 | 0.56 / 0.00 | 0.39 / 0.00 |
    | | 8 vops | 1.00 / 1.00 | 0.74 / 0.01 | 0.60 / 0.00 | 0.42 / 0.00 |
    | lists starting anywhere in the first 16 places | 128 vops | 1.00 / 1.00 | 0.99 / 0.83 | 0.80 / 0.00 | 0.58 / 0.00 |
    | | 8 vops | 0.99 / 0.90 | 0.84 / 0.08 | 0.67 / 0.00 | 0.50 / 0.00 |
    | a GRU of the same size (finding 30) | | 1.00 / 1.00 | 0.57 / 0.00 | 0.50 / 0.00 | 0.35 / 0.00 |

    The trace of the third row is a loop: after three ticks of setting up, one vop (number
    75) runs on every tick while the read pointer and the write pointer step along the list
    together, 0 0 1 1 2 2 3 3 4 ... On a list of 12, half as long again as any it trained
    on, the same vop keeps running and both pointers walk on to place 11: 83% of such lists
    are copied whole. The machine trained with lists always at the start also settles on
    one vop and steps its write pointer, but past place 8 the pointer jumps about: those
    addresses were never seen. A small bank was not what forced the loop; it loops with 128
    vops too. At 16 and 24 it still fails whole lists.
32. **Where the copy loop fails on long lists it is not losing its place step by step; it
    is confusing places whose addresses look alike.** [likely: `python -m vvm2.drift`, two
    trainings] On a list of 16 the middle is 0.99 right and the two ends fail (0.47 to 0.76).
    The neighbouring item is written 1 time in 8, which is chance, so the pointers have not
    slipped. The wrong item is the one 12 or 13 places away (half the time at the start of
    the list, against 1 in 8 by chance). The position code is six sine waves, and with the
    rates first chosen, places 12 apart are as alike (0.28) as places 4 apart (0.29). The
    pointers are also smeared over more slots on longer lists: 3 on a list of 8, 6 on 16,
    8 on 24. A search over rates lowers the worst far likeness only from 0.33 to 0.25:
    twelve numbers of position is a small address space.

## Exact pointers (`vvm2/core.py` as it is now)

33. **With pointers that are places moved by steps, copy and reverse are learned as loops
    and carry to lists twelve times longer than any trained on.** [likely: one seed each,
    3,000 steps, two trainings agree; `scripts/tape.sh copy`, `scripts/tape.sh reverse`]
    Trained on lengths 2 to 8; items right / whole lists right:

    | task | how it is run | 8 | 12 | 16 | 24 | 48 | 96 |
    |---|---|---|---|---|---|---|---|
    | copy | as trained (steps blended) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.95 / 0.34 | 0.58 / 0.00 |
    | copy | with whole steps | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
    | reverse | as trained | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.98 / 0.60 | 0.67 / 0.00 |
    | reverse | with whole steps | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
    | copy | a GRU of the same size | 1.00 / 0.99 | 0.59 / 0.00 | 0.50 / 0.00 | 0.36 / 0.00 | 0.24 / 0.00 | |
    | reverse | a GRU of the same size | 0.99 / 0.95 | 0.68 / 0.00 | 0.55 / 0.00 | 0.41 / 0.00 | 0.27 / 0.00 | |

    The trace is a program: a few ticks of setting up (for reverse, the read pointer goes to
    the last input slot), then one vop on every tick, the read pointer stepping one way and
    the write pointer the other. As trained, a pointer is about 0.9 on its one slot and the
    blur adds up over a long list (this is error building up along the loop, which finding
    32 ruled out for the earlier form). Run with whole steps, every pointer taking its
    likeliest step outright, there is no blur and no length at which it fails. Half the
    weights of the address-vector form (0.49 million).
34. **The lights game is not yet a routine in this form.** [hint: one seed] 12,000 steps:
    1.00 / 1.00 at 6 lights, 0.80 / 0.07 at 8, 0.61 at 12; a GRU 0.79 / 0.12 and 0.61. Its
    pointers are about half exact, and run with whole steps it falls to 0.47: what it
    learned depends on the blur.
35. **A move by content is built and does not yet learn a lookup.** [hint: one seed, 4,000
    steps, `scripts/tape.sh lookup`] "Seek" was added to a pointer's steps: go to the slot
    whose contents best match a vector the vop supplies. Find a value by its key among 2 to
    6 pairs, tested on 6, 12 and 24: the machine 0.38, 0.28, 0.21 (chance 0.125) and 0.15
    run with whole steps; a GRU of the same size 1.00, 1.00, 0.82. The trace shows pointers
    seeking on most ticks with no routine: it never finds "seek the key, step to the slot
    beside it, carry that to the output". Copy and reverse are unharmed by the extra move
    (run with whole steps: all items right at every length to 96; whole lists 0.95 to 1.00).
36. **Lookup by key is learned, and carries to four times as many pairs; it had only been
    undertrained.** [hint: one seed each, `scripts/tape.sh lookup 16000`, `... 8000 twin=1`]
    Trained on 2 to 6 pairs; share of answers right at 6, 12 and 24 pairs (chance 0.125; a
    GRU of the same size 1.00, 1.00, 0.82):

    | training | as trained | run with whole steps |
    |---|---|---|
    | 4,000 steps | 0.38, 0.28, 0.21 | 0.15, 0.13, 0.12 |
    | 16,000 steps | 1.00, 1.00, 1.00 | 0.78, 0.77, 0.75 |
    | 8,000 steps, one pair first and growing to six | 0.39, 0.25, 0.21 | 0.14, 0.14, 0.12 |
    | 8,000 steps, with a built-in step "go to another slot like the one I am on" | 1.00, 1.00, 1.00 | 0.98, 0.97, 0.98 |
    | 8,000 steps, both of those | 1.00, 1.00, 1.00 | 0.65, 0.62, 0.68 |

    With the built-in step the trace is the routine one would write by hand: on the first
    tick the read pointer goes from the asked-for key to its twin, on the second it steps
    to the slot beside it, and the value there is carried to the output. That is a move by
    content followed by a move by position. Without the built-in step it gets every answer
    right as trained but leans on blended pointers (0.77 with whole steps). The curriculum
    of data, as tried here, did not help.
37. **Copy and reverse hold on three seeds, and each seed writes a different program.**
    [solid: 3 seeds each, `scripts/tape.sh copy 3000 seed=N`] Run with whole steps, every
    item and every whole list is right at every length from 8 to 96 on all three seeds of
    both tasks (trained on 2 to 8). One copy starts at the front and steps forward; another
    sends both pointers to the far end and copies backwards. One reverse reads from the end
    and writes from the start; another reads from the start and writes from the end; a
    third makes a false start of two items, goes back to the beginning and does it properly.
38. **The current machine fits ordinary curves as well as an MLP, but only with blurred
    pointers, unless it is also trained on the exact run.** [solid: 3 seeds,
    `scripts/curves.sh`, docs/results-2026-11-27-curves-control.md] Bank of 16, 4 ticks,
    the same weights as the MLP beside it.

    | task | as trained | the same weights, whole steps | trained on the whole-step run too, whole steps | MLP |
    |---|---|---|---|---|
    | xor, noisy test points | 0.972 | 0.791 | 0.942 | 0.998 |
    | a sine, squared error | 0.00002 | 0.452 | 0.00002 | under 0.000005 |
    | two spirals | 1.000 | 0.555 | 0.985 | 1.000 |

    A curve gives the pointers nothing to do, so they stay blurred and the fit leans on the
    blur. Running every batch twice, once as usual and once with whole steps, and adding
    the two losses (`mix=1`) closes the gap. Nothing in the machine changes.
39. **It balances a pole, trained with no labels, and brings nothing an MLP does not.**
    [solid for "it balances": 6 of 6 runs, `scripts/control.sh`] Trained by an evolution
    strategy with whole steps, forward only: 500 of 500 steps on 100 of 100 test episodes
    every time. It took about 9,700 episodes of experience against 3,300 for an MLP of the
    same weights and 5,300 for a GRU (settings not tuned for any of them). It does half the
    multiply-adds a step of that MLP. The trained machine parks its pointers and runs one
    vop: a reflex with a little memory in the pc slot. Scratch is never used.
40. **The lights game fits in the machine: two vops written by hand play any length and any
    hold.** [solid: exact, `python -m vvm2.simon_hand`] One vop stores the light and steps
    the write pointer while the signal says watch, and on "play" rewinds the read pointer,
    sends the write pointer to the button and jumps the pc; the other copies scratch to the
    button and steps. Every game right at 6 to 48 lights and 1 to 8 ticks a light. So
    nothing is missing from the design, the conditional jump included; what fails is
    training finding the routine. Trained from scratch it puts each light at a fixed
    landmark (a region end) with a different vop each tick, which is exact for three or
    four lights and cannot grow. The earlier form of the task with one tick a light was
    also unfair: it wanted the first answer on the tick the machine first saw "play",
    before a pointer could move; `vvm2/simon.py` gives one tick of reaction time.
41. **A ladder of small algorithms: the machine carries a plain loop and stops where a
    pointer must wait or turn by what it reads.** [hint: one seed a row but shift,
    `scripts/algos.sh`, docs/results-2026-11-27-algorithms.md] Trained on lengths 2 to 8, a
    bank of 128; share of whole answers right.

    | task | length 8, as trained | length 64, whole steps | length 64, whole steps, no move by content | GRU, 8 / 64 |
    |---|---|---|---|---|
    | shift by one | 1.00 | 0.10 | 1.00, 0.76, 1.00 (3 seeds) | 1.00 / 0.00 |
    | each item twice | 1.00 | 0.93 | | 0.97 / 0.00 |
    | keep the items of one kind | 0.75 | 0.00 | 0.00 | 0.99 / 0.00 |
    | count the items of one kind | 0.96 | 0.00 | 0.00 | 1.00 / 0.00 |
    | the list, then the list again | 0.01 | 0.00 | 0.00 | 0.35 / 0.00 |
    | parity | 0.95 | | 0.55 | 1.00 / 1.00 |

    "Each item twice" is a real loop of two vops (the write pointer steps every tick, the
    read pointer every other tick). Shift learns the right loop but starts it with a move
    by content that lands in the wrong place once steps are whole; with that move held off
    it is exact. Keep, count, the doubled list and parity are not learned as routines:
    each needs a pointer to step or not by the item just loaded, or to turn at the end of
    a list. Parity is the one task a GRU carries to length 64 and the machine does not.
42. **A conditional jump is learned; it is not the weak spot.** [solid for "learned": 3 of
    3 seeds] A signal slot says copy or reverse. Right at trained lengths on every seed.
    With the signal dark after the first tick the pc carries it and a different loop runs
    for each signal; at length 64 with whole steps 0.99 to 1.00 of items are right on all
    three seeds. With the signal left lit, two seeds of three never change the pc: one vop
    reads the signal each tick and steps the write pointer forward or back by it.
43. **The move by content is used as a blur in training and often breaks the exact run.**
    [hint] Shift and the copy-or-reverse task both become exact at length 64 when the
    move by content is held off (finding 41, 42), and the lookup that worked best used the
    built-in "another slot like this one" form, not a vector the vop supplies (finding 36).

