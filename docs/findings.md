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

