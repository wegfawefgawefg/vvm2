# Images, through pointers that move on a grid and zoom

2026-11-27. Code: `vvm2/image.py` (the device, the classifier), `vvm2/look.py` (the look
inside), `vvm2/shapes.py` (the shapes, drawing, generating), `scripts/image.sh`. The core is
not touched.

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
