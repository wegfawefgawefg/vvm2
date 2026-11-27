# Images, through pointers that move on a grid and zoom

2026-11-27. Code: `vvm2/image.py` (the device, the classifier), `vvm2/look.py` (the look
inside), `vvm2/shapes.py` (the shapes, drawing, generating), `scripts/image.sh`. Nothing in
the machine was changed for this work; `vvm2/core.py` was swapped part way through for the
faster one from main (same numbers), as the coordinator said to. The predictions below are
as committed; the results follow them.

## Predictions (written and committed before any scoring run)

The only thing run before these was `python -m vvm2.image time`, which scores nothing: with
the settings below the machine has 2.26 million weights and a training step takes about a
third of a second on 4 threads.

Settings predicted for: 12 ticks, 4 read pointers, a bank of 128 vops (4 fetched), 8 scratch
slots, 3,000 steps of batch 128, images moved by up to 2 pixels in training, an MLP of the
same number of weights trained the same way.

1. **MNIST.** The machine as trained: test 0.975 (somewhere in 0.96 to 0.985). The MLP:
   0.985 to 0.99, so the MLP wins. Run with whole steps the machine is much worse: 0.75
   (0.5 to 0.9), because training lets a pointer be a blend over many patches and it will
   use that as pooling; pointers exact to 0.5 to 0.7. Run with whole steps it looks at 15 to
   30 of the 66 slots. Most of the bank is used (more than half the vops), and the pick
   changes with the input on 0.3 to 0.6 of ticks.
2. **Kuzushiji:** machine 0.88 (0.84 to 0.92), MLP 0.93 to 0.95. **Fashion:** machine 0.86
   (0.84 to 0.88), MLP 0.89 to 0.90. Whole steps cost both about what they cost MNIST.
3. **Teaching with whole steps** (further training in the running mode; see `tune`) brings
   the whole-steps score to within 0.02 of the as-trained score.
4. **Shapes, naming.** At trained sizes and places both are above 0.97. On large sizes:
   both 0.6 to 0.85, no clear winner. On the right third: the MLP 0.45 to 0.7 (chance 0.33),
   the machine a little better (0.5 to 0.8) and not solved. The machine does not learn a
   "find the ink, then look" routine that would make place not matter.
5. **The look inside.** It zooms: the fine level is reached on nearly every run, because the
   whole-image slot alone cannot give 0.9. The path is mostly the same for every image
   (fewer than 30% of pointer places differ from the commonest place at that tick). The ink
   under the pointers is above the average patch's, but about the same as the ink at the
   same places on other images (ratio 1.0 to 1.2): a fixed path through the middle, not a
   move toward the ink.
6. **Shapes, drawing** (if run): a blurred shape at about the right place; pixel error 1.5
   to 3 times the MLP's.
7. **Generating Fashion** (if run): recognisable as its class, blurrier than the MLP given
   the same class and style numbers; pixel error above the MLP's.

### A further prediction, written after the first two runs and before the run it is about

Seen by then (one seed each): MNIST at 2,000 steps 0.951 as trained and 0.29 run with whole
steps; naming shapes 1.000 as trained and 0.59 with whole steps, 0.65 after 500 steps taught
with whole steps. So prediction 3 is already wrong for shapes.

8. **Whole steps on half the training passes** (`mix=0.5`; such a pass teaches values and
   gives the choice of step no gradient), shapes, one seed: the score run with whole steps
   rises to 0.8 to 0.9 and the as-trained score stays above 0.98. Low confidence: nothing in
   it teaches the machine which step to take when steps are whole.

## The answers, a sentence each

1. **The device** is built as asked and the core's workings are untouched: 66 input slots
   (the whole image, 16 coarse patches, 49 fine patches), six steps, no encoder, the answer
   read from one output slot.
2. **Classifying:** as trained the machine reaches 0.92 to 0.93 on MNIST in 1,000 steps (three
   seeds) and 0.95 in 2,000 (one seed), below an MLP of the same weight count (0.98 to
   0.99), and run with whole steps it falls to about 0.3; on Kuzushiji 0.73 against the
   MLP's 0.91 and on Fashion 0.78 against 0.85 (one seed each, 1,000 steps).
3. **Naming shapes** is the good news: as trained the machine is right on 0.98 to 1.00
   of shapes larger than any it trained on and on 0.98 to 0.99 of shapes centred in the
   third of the canvas where it never saw a centre, on three seeds, where the MLP gets 0.84
   to 0.86 and 0.37 to 0.42 (chance 0.33). Run with whole steps the same machines get 0.40
   to 0.59; trained with half the passes themselves run with whole steps, 0.83 on one
   seed and 0.54 on another, so that recipe is not yet something to lean on.
   Drawing a shape (task b) was not run.
4. **The look inside:** it does move toward the ink, and it does it by seeking (the move by
   content), not by walking the grid; zooming is a small share of its steps on shapes and a
   fixed habit on MNIST (whole image, then the middle coarse patch, then the fine patch
   under it), not a choice made from what it sees.
5. **Generating** was not run (cut for time at the coordinator's word); the code is there
   and runs end to end.

The thing to know before reading the tables: **as trained on images, a pointer is not a
place.** It is a blend over about ten slots, and `seek` as trained is a soft match against
every slot at once. That is what scores well. The machine "run with whole steps" is a
different and much weaker reader, and teaching it after the fact does not close the gap.

## What was run

The machine everywhere: `Core(moves=6)`, 4 read pointers, 12 ticks, a bank of 128 vops with
4 fetched, 8 scratch slots; 2,261,808 weights and about 1.08 million multiply-adds an image.
The MLP beside it has two hidden layers and the same weight count, so 2.26 million
multiply-adds: at equal weights it has twice the machine's compute. An MLP matched on
compute (half the size) was not run, nor a GRU or a transformer. Both models: batch 128,
AdamW, learning rate 3e-3 on one cycle, weight decay 0.05; on the three data sets training
images are moved by up to 2 pixels.

`vvm2/core.py` was replaced part way through by the faster one from main, as the
coordinator said to (same numbers: a saved machine scored 0.9207 and 0.3306 under both).
Runs marked * used it; the others used the core as it was at the start.

Columns: "pointers exact" is the mean share of a pointer on its likeliest slot (1.0: every
pointer on one slot). "Slots looked at" is how many different input slots the read pointers
are on over a run, counting each pointer's likeliest slot only, so it is given only for the
rows run with whole steps. Bank columns are by the best-matching vop of each tick, over
2,000 test images: share of the 128 vops ever used, entropy of use (1.0: all equally), the
top vop's share, and the share of runs whose vop at a tick is not that tick's commonest.

### Classifying

| data, steps, seed | model | train | test | pointers exact | slots looked at (whole, coarse, fine) | vops used | entropy | top vop | pick changes with input |
|---|---|---|---|---|---|---|---|---|---|
| MNIST, 1,000, seed 0 | vvm as trained | 0.921 | 0.925 | 0.38 | | 0.20 | 0.43 | 0.31 | 0.45 |
| | vvm, whole steps | 0.322 | 0.318 | 1.00 | 9.4 (1.0, 3.0, 5.4) | 0.66 | 0.54 | 0.19 | 0.62 |
| | mlp | 0.983 | 0.981 | | | | | | |
| MNIST, 1,000, seed 1 | vvm as trained | 0.922 | 0.921 | 0.47 | | 0.30 | 0.40 | 0.50 | 0.41 |
| | vvm, whole steps | 0.346 | 0.331 | 1.00 | 6.5 (1.0, 1.9, 3.6) | 0.74 | 0.56 | 0.33 | 0.59 |
| | mlp | 0.984 | 0.982 | | | | | | |
| MNIST, 1,000, seed 2 * | vvm as trained | 0.930 | 0.932 | 0.42 | | 0.23 | 0.46 | 0.25 | 0.48 |
| | vvm, whole steps | 0.336 | 0.334 | 1.00 | 4.9 (1.0, 1.7, 2.2) | 0.74 | 0.60 | 0.17 | 0.65 |
| | mlp | 0.985 | 0.982 | | | | | | |
| MNIST, 2,000, seed 0 | vvm as trained | 0.952 | 0.951 | 0.37 | | 0.22 | 0.50 | 0.18 | 0.53 |
| | vvm, whole steps | 0.287 | 0.291 | 1.00 | 8.8 (1.0, 2.7, 5.1) | 0.72 | 0.62 | 0.14 | 0.63 |
| | then 600 steps taught with whole steps: as trained | 0.818 | 0.809 | 0.39 | | 0.20 | 0.49 | 0.20 | 0.51 |
| | the same, whole steps | 0.561 | 0.560 | 1.00 | 8.1 (1.0, 2.5, 4.6) | 0.62 | 0.58 | 0.14 | 0.65 |
| | mlp | 0.989 | 0.986 | | | | | | |
| Kuzushiji, 1,000, seed 0 | vvm as trained | 0.867 | 0.735 | 0.47 | | 0.34 | 0.48 | 0.27 | 0.57 |
| | vvm, whole steps | 0.504 | 0.428 | 1.00 | 5.8 (1.0, 1.9, 2.9) | 0.65 | 0.54 | 0.28 | 0.63 |
| | mlp | 0.973 | 0.914 | | | | | | |
| Fashion, 1,000, seed 0 | vvm as trained | 0.783 | 0.776 | 0.44 | | 0.17 | 0.42 | 0.26 | 0.55 |
| | vvm, whole steps | 0.520 | 0.522 | 1.00 | 7.1 (1.0, 2.4, 3.8) | 0.65 | 0.50 | 0.28 | 0.56 |
| | mlp | 0.859 | 0.852 | | | | | | |

The machine is still climbing when these runs stop (MNIST 0.925 at 1,000 steps, 0.951 at
2,000); the MLP is near its ceiling. Nothing here says where the machine's ceiling is.

Bank use: the pick changes with the input on about half the ticks, so it is not the same
vops whatever the input. But as trained a fifth to a third of the bank does all the work
beside a score short of the MLP's, which is the pattern AGENTS.md asks to be flagged.

### Naming shapes

Three classes (circle, square, triangle; filled or outlined is not asked). Trained on
half-widths 3 to 7 pixels with the centre in the left two thirds. Chance is 0.33.

| seed, steps | model | trained sizes and places | large (half-width 8 to 10) | centre in the right third |
|---|---|---|---|---|
| 0, 1,500 | vvm as trained | 1.000 | 0.996 | 0.992 |
| | vvm, whole steps | 0.592 | 0.583 | 0.582 |
| | then 500 steps taught with whole steps: as trained | 0.957 | 0.973 | 0.922 |
| | the same, whole steps | 0.653 | 0.660 | 0.635 |
| | mlp | 0.997 | 0.861 | 0.419 |
| 1, 1,000 | vvm as trained | 1.000 | 0.986 | 0.979 |
| | vvm, whole steps | 0.398 | 0.399 | 0.396 |
| | mlp | 0.993 | 0.847 | 0.385 |
| 2, 1,000 | vvm as trained | 0.999 | 0.981 | 0.979 |
| | vvm, whole steps | 0.493 | 0.521 | 0.498 |
| | mlp | 0.992 | 0.842 | 0.365 |
| 0, 1,500, half the passes with whole steps | vvm as trained | 0.999 | 0.981 | 0.994 |
| | vvm, whole steps | 0.834 | 0.874 | 0.800 |
| 1, 1,500, half the passes with whole steps * | vvm as trained | 0.997 | 0.984 | 0.992 |
| | vvm, whole steps | 0.537 | 0.515 | 0.518 |

Bank and pointers, seed 0, plain run: pointers exact 0.35; run with whole steps 12.4 input
slots looked at a run (0.7 whole, 3.6 coarse, 8.1 fine); vops used 0.09 as trained and 0.84
with whole steps; entropy 0.43 and 0.57; top vop 0.18 and 0.16; pick changes with input 0.39
and 0.57. With half the passes whole: 13.1 slots (0.7, 3.9, 8.5); vops used 0.08 and 0.54;
entropy 0.37 and 0.46; top vop 0.31 and 0.25; pick changes 0.27 and 0.37. A tenth of the
bank is enough for this task as trained; the score is at the top, so that is not a fault.

## The look inside

`scripts/image.sh look mnist`, `scripts/image.sh look shapes`, `scripts/image.sh look shapes
which=mix`. Counts are over 2,000 test images, seed 0 (MNIST: the 2,000-step machine; the
half-whole shapes machine: the seed that scores 0.83 with whole steps, not the one at 0.54).

**Where the pointers start.** The start is the core's learned blend over six region ends,
and it stays close to even (0.17 to 0.39 on the likeliest end). On MNIST the read pointers'
likeliest start is the whole-image slot. On shapes it is not: they start on the last fine
patch (the bottom right corner) and on scratch. So "it starts with the coarse view" is not
something the device can promise; see what is missing.

**As trained, a read pointer is a blend.**

| | MNIST | shapes | shapes, half the passes whole |
|---|---|---|---|
| slots a read pointer is spread over (exp of its entropy) | 9.6 | 12.0 | 13.1 |
| share of the read pointers on: the whole image | 0.28 | 0.09 | 0.08 |
| the coarse level | 0.08 | 0.08 | 0.10 |
| the fine level | 0.16 | 0.27 | 0.30 |
| scratch and output | 0.48 | 0.56 | 0.52 |
| commonest steps, by share of the blend | to first input slot 0.19, seek 0.12, zoom in 0.10 | to output 0.19, seek 0.17, to scratch 0.15 | seek 0.27, to output 0.20, to scratch 0.13 |
| zoom in, zoom out | 0.10, 0.07 | 0.06, 0.04 | 0.05, 0.03 |
| right, left, down, up together | 0.14 | 0.20 | 0.16 |
| ink under the fine part of the blend: this image | 0.156 | 0.165 | 0.167 |
| the same places on another image | 0.138 | 0.077 | 0.086 |
| a whole image's mean | 0.121 | 0.078 | 0.078 |

**Does it move toward the ink?** On shapes, yes: the fine-level part of the blend sits on
twice the ink it would find at the same places on another image (0.165 against 0.077), and
a quarter of the blend is somewhere other than where it is on the average image. On MNIST,
barely (0.156 against 0.138; digits are all in the middle, so a fixed look at the middle
already finds ink).

**How?** By `seek`. As trained, seeking is a soft match of a vector the vop supplies against
every slot in memory, so one seek is a look by content over the whole image at all three
levels at once. It is the commonest step onto the image on shapes. It knows nothing of where
a patch is, which is the likely reason a shape in a new place or of a new size is named as
well as a trained one (not tested by taking seek away).

**Does it zoom?** Zooming is used, but not as "look at the coarse view, then choose where to
zoom". On MNIST run with whole steps every run reaches the coarse and the fine level, and
in the traces the usual route is the device's fixed chain: whole image, zoom in to the
middle coarse patch (C11), zoom in to the fine patch under it (F22), back to the whole
image. On shapes zoom-in is 2 to 3% of whole steps and seek is 61 to 67%. The four grid
steps are a few percent each everywhere. So the grid and the zoom are there and are mostly
passed over for the move by content.

**Run with whole steps** the same machine takes each pointer's likeliest step and seeks to
the single best match:

| run with whole steps | MNIST | shapes | shapes, half the passes whole |
|---|---|---|---|
| looks by read pointers on: whole, coarse, fine, off the image | 0.44, 0.13, 0.19, 0.23 | 0.04, 0.20, 0.43, 0.33 | 0.03, 0.22, 0.51, 0.24 |
| places not the commonest for that pointer and tick | 0.46 | 0.71 | 0.75 |
| ink under the read pointers: this image, another image | 0.200, 0.188 | 0.224, 0.122 | 0.231, 0.123 |
| commonest steps | to first input slot 0.30, zoom in 0.19, seek 0.15 | seek 0.61 | seek 0.67 |

A trace, shapes with half the passes whole, a square (class 1), run with whole steps (W the
whole image, C12 coarse row 1 column 2, F35 fine row 3 column 5, s scratch, out the output
slot; under each place the step then taken):

```
tick             0        1        2        3        4        5        6        7        8        9       10       11
vop             24       14       49       78       27       78       20       20       20       20       20       20
read 1         F66        W      F30      F20      F33      F20      F33      F43      C11      out      C11      F33
  then        seek     seek       up     seek     seek     seek     seek     seek  to-out0     seek     seek  to-outN
read 2         F66      C10      F50      F40      F30      F40      F33      F30      F40      C22      C12      F24
  then        seek     seek       up     seek     down     seek     seek     seek     seek     seek     seek  to-out0
read 3          s0        W      C10      F20      F33      C21      F20      F50      C21      F50      F51      out
  then        seek     seek     seek     seek     seek     seek     seek     seek     seek     seek     seek     seek
read 4          s0      F33      C22      C22      out      out      out      out      out      out      C20      out
  then        seek     seek     stay  to-outN  to-out0  to-outN  to-outN  to-outN  to-outN     seek  to-outN  to-outN
write          F66      C20      F33      C21      F30      F20      F30      F20      F33      out      out      out
  then        seek     seek     seek     seek     seek     seek     seek     seek  to-out0  to-out0  to-out0  to-outN
```

The read pointers jump from the corner onto the square's left edge and corners (F20 to F50,
F33) and hop among them by seeking; one vop (20) runs the last six ticks. The write pointer
spends eight ticks on input slots, where a write does nothing, and reaches the output slot
only at tick 9.

An MNIST trace (a 7, called 2), run with whole steps: all four read pointers start on the
whole image; at tick 1 they are on F34, F34, F34 and C11 (three by seeking, one by zooming
in); after that they cycle whole image, C11, F22 with the odd seek. The look prints three
traces in full for each machine.

Pictures: `docs/img/look-shapes-mix.png` (whole steps: boxes on the shape's edges wherever
the shape is; the clearest one), `docs/img/look-shapes.png`, `docs/img/look-mnist.png`
(whole steps), and `docs/img/look-mnist-blend.png`, `docs/img/look-shapes-blend.png`,
`docs/img/look-shapes-mix-blend.png` (as trained: where the blend is, tick by tick).

## The predictions against what happened

1. MNIST as trained: low (0.951 at 2,000 steps against a predicted 0.975 at 3,000, which
   was not run). MLP: right. Whole steps: wrong, far worse than predicted (about 0.3 against
   0.75). Pointers exact 0.37 to 0.47 against 0.5 to 0.7. Slots looked at with whole steps
   5 to 9 against 15 to 30. Bank use as trained 0.20 to 0.30 of vops, not "more than half";
   pick changes with input: in range.
2. Kuzushiji and Fashion: both models well under the predictions, which were for 3,000
   steps; at 1,000 the machine is at 0.73 and 0.78 and the MLP at 0.91 and 0.85. The MLP
   wins both, as predicted. Whole steps cost less than on MNIST (0.43 and 0.52), not the
   same.
3. Teaching with whole steps afterwards: wrong. MNIST 0.56 and shapes 0.65, nowhere near
   the as-trained scores, and on MNIST it wrecked the as-trained score (0.951 to 0.809).
4. Shapes: wrong, in the machine's favour. Predicted no clear winner on large shapes and
   "not solved" on place; the machine as trained is far ahead of the MLP on both.
   Predicted no find-the-ink routine; there is one, by seeking.
5. The look: "a fixed path through the middle, not a move toward the ink" is about right
   for MNIST and wrong for shapes. "It zooms because it has to": half right; it reaches the
   fine level mostly by seeking.
6. and 7. Not run.
8. Half the passes whole: right on seed 0 (0.834 with whole steps, 0.999 as trained).
   Wrong on seed 1 (0.537 with whole steps, 0.997 as trained). The recipe does not hold
   across seeds.

## What is suspect

- **Seeds and lengths are thin.** MNIST is three seeds at 1,000 steps and one at 2,000;
  Kuzushiji and Fashion one seed at 1,000; shapes one seed at 1,500 and two at 1,000;
  teaching with whole steps afterwards is one seed; half the passes whole is two, and they
  disagree (0.83 and 0.54). Only MNIST at 1,000 steps and the shapes scores as trained have
  three seeds. The look inside is one seed of each machine, and for the half-whole machine
  it is the seed that did well. The CPU was
  shared four ways and a training step took 0.4 to 1.9 seconds, so runs were cut.
- **The machine is undertrained** on the data sets and the MLP is not; the gap at equal
  steps overstates the gap at convergence by an unknown amount. Kuzushiji shows the
  machine fitting the training images (0.87) well ahead of the test images (0.73).
- **The as-trained machine on images is soft attention.** A blend over ten slots through a
  soft match by content is close to what an attention layer does. The shapes result is
  real, but it is a result about that, not about exact pointers walking a grid.
- **Why shapes carry is read off traces and counts, not tested.** No run took seek away or
  held the pointers to the grid steps.
- **The held-out place test is by the shape's centre.** Trained shapes do reach into the
  right third (a wide shape centred at column 18 covers up to column 25); what is held out
  is a centre there. The MLP's 0.42 is what a net with no notion of place does; a
  convolutional net would carry. No such baseline was run.
- **Large shapes are tested at trained places only**, and right-third shapes at trained
  sizes only; both at once was not tried.
- **The coarse level is padded** on the right and bottom (28 is not a multiple of 8), so
  its last row and column are half empty, and "zoom in" from a coarse patch always lands on
  the upper left of its four fine patches.
- **"Run with whole steps" train scores** are on the first 10,000 training images.
- **The half-whole recipe** flips the core into its running mode for a pass with gradients
  left on (`ImageVM.forward`, `mixed`). It changes nothing in `core.py`, but it uses the
  core in a way it was not written for; the owner should say if that counts as leaving the
  machine alone.

## What is missing from the machine (nothing was changed)

1. **No way to learn which step to take when steps are whole.** Training blends; running
   takes the likeliest. On lists the blend sharpens by itself (finding 33). On images it
   does not (pointers exact 0.35 to 0.47), because a blurred pointer is useful there. A
   whole step passes no gradient to the choice, and the core hands a task no blur to charge
   for (`exact` and the trace are detached). Teaching values under whole steps, after or
   during training, is all a task can do from outside; it gets shapes to 0.54 to 0.83 and
   MNIST to 0.56.
2. **`seek` is two different things** in training and in running: a soft match over every
   slot, and a jump to the one best match. The first is strong and is not a pointer.
3. **The start cannot be set by the device.** Each pointer starts on a learned blend of six
   region ends; the device cannot say "start on the whole-image slot".
4. **Device steps reach input slots only.** `go` covers the input region; scratch and
   output are lines. A canvas of 49 output slots cannot be given right, left, down and up.
5. **A pointer cannot be read.** The machine cannot ask where a pointer is; on a canvas the
   world has to lay out a ruler (slots holding their own row and column) for it to walk.

## Not done

- Task b (drawing a shape) and item 5 (generating clothes and bags). `vvm2/shapes.py` has
  both (`Painter`: the request and a 49-slot ruler in, a 49-slot canvas out, one tick a
  patch plus 4, an MLP of the same size beside it; for generating, a linear map of 784 to 8
  outside the machine makes the style numbers, as old/mm.py did, and everything else is the
  machine). They run end to end in `scripts/image.sh --quick` and have never been trained
  past 10 steps. No numbers.
- More than one seed of Kuzushiji and Fashion; 3,000-step runs; a third seed of shapes at
  1,500 steps (seeds 1 and 2 are 1,000).
- Baselines other than the equal-weights MLP: none matched on compute, no GRU, no
  transformer, no convolutional net for the place test.
- The half-whole recipe on MNIST, and plain and half-whole shapes runs of the same length
  on the same seed beyond seed 0 (seed 1 is 1,000 steps plain and 1,500 half-whole).
- Any sweep of ticks, read pointers or bank size.

## The script lines

```
scripts/setup.sh
scripts/get_mnist.sh; scripts/get_data.sh kmnist; scripts/get_data.sh fashion
scripts/image.sh time
scripts/image.sh class mnist 1000 seed=0        (also seed=1, seed=2)
scripts/image.sh class mnist 2000 seed=0 tune=600
scripts/image.sh class kmnist 1000 seed=0
scripts/image.sh class fashion 1000 seed=0
scripts/image.sh class shapes 1500 seed=0 tune=500
scripts/image.sh class shapes 1000 seed=1       (also seed=2)
scripts/image.sh class shapes 1500 seed=0 mix=0.5 mlp=0     (also seed=1)
scripts/image.sh look mnist seed=0              (after the 2,000-step run)
scripts/image.sh look shapes seed=0
scripts/image.sh look shapes seed=0 which=mix
scripts/image.sh --quick
```

`IMAGE_THREADS=1 scripts/image.sh ...` runs on one thread (4 unless said); the later runs
here did. A `class` run saves its machine in `out/image/`, and `look` reads it, so a later
`class` run of the same task and seed replaces what `look` sees.

## Proposed edits (not made)

**docs/findings.md**, three new findings:

38. **On images the machine as trained is a soft reader, and run with whole steps it
    fails.** [likely: MNIST three seeds at 1,000 steps and one at 2,000; one seed each of
    Kuzushiji and Fashion; `scripts/image.sh class mnist 2000 tune=600`] Through a device
    that lays an image out at three levels with grid and zoom steps: MNIST 0.92 to 0.93 at
    1,000 steps and 0.95 at 2,000, an MLP of the same weights 0.98 to 0.99; run with whole steps
    about 0.3. Kuzushiji 0.73 and Fashion 0.78 at 1,000 steps (MLP 0.91 and 0.85). A pointer
    is spread over about ten slots (exact to 0.35 to 0.47). 600 steps of teaching with
    whole steps afterwards: 0.56 on MNIST, and the as-trained score falls to 0.81.
39. **Naming shapes carries to sizes and places not trained on, by seeking.** [solid for
    the as-trained scores: 3 seeds; the reason is read off traces, not tested;
    `scripts/image.sh class shapes 1500`] The machine as trained names 1.00 of fresh shapes,
    0.98 to 1.00 of shapes larger than any trained on and 0.98 to 0.99 of shapes centred in
    the right third; an MLP of the same weights 0.99 to 1.00, 0.84 to 0.86 and 0.37 to 0.42
    (chance 0.33). The read pointers seek onto the shape's edges wherever it is; the grid and zoom steps are a few percent of steps. As
    trained the seek is a soft match over all slots, so this is attention by content, not a
    walk. Run with whole steps the same machines get 0.40 to 0.59.
40. **Running some training passes with whole steps helps the whole-step run on one seed
    and little on another.** [hint, and it does not repeat: two seeds, shapes;
    `scripts/image.sh class shapes 1500 mix=0.5`] With half the passes whole (values
    taught, no gradient to the choice of step), the score run with whole steps is 0.83 on seed 0 (plain
    training 0.59; held out 0.87 large and 0.80 right third, above the MLP on place) and
    0.54 on seed 1 (plain training, 1,000 steps, 0.40). The as-trained scores are unharmed
    (1.00, 0.98, 0.99). Teaching with whole steps only afterwards: 0.65 on seed 0. Better
    than nothing and not a fix.

**docs/ideas.md**, new ideas:

63. **A gradient for the choice of step under whole steps:** take the whole step going
    forward and pass the gradient back through the blend, as a switch on the core. It is
    the missing piece behind findings 34, 38 and 40.
64. **Let a task charge for blur:** return how exact the pointers are with its gradient, so
    a world can add "be exact" to its loss without touching the tick.
65. **Decide what seek is.** As trained it is a soft match over every slot (attention); run,
    it is one jump. Either make the trained form as narrow as the run form, or own that the
    machine has attention and say so in the design page.
66. **A share of training passes run with whole steps** helped on one seed of shapes and
    not much on another (finding 40). Worth one try on the lights game and on lookup, where
    whole-step scores trail the trained ones, before idea 63 is built.
67. **A device may fix where pointers start** and may supply steps for scratch and output
    (a canvas with right, left, down, up), not only for input.
68. **A slot that reports a pointer's place** (beside "a slot that can hold a place" in Not
    built): without it the world must lay out a ruler for the machine to know where it is.
69. **Seek within a level, or seek that prefers nearby slots**, so that finding ink and then
    walking from it are one routine; at present seeking makes the grid unnecessary.
