# Cellular automata, learned on small boards and run on larger ones

2026-11-27. Code: `vvm2/automata.py`, `scripts/automata.sh`. No task here changes the core;
`vvm2/core.py` was updated once from main (a faster tick with the same numbers).

The predictions at the end were committed before the runs they are about (d2c0fb8, 270226e
and the two prediction commits after them) and are not edited. Scores are share of cells
right / share of whole boards right, on 300 fresh boards a size. "Whole steps" is the same
trained machine with `core.hard = True`.

## The answer, one sentence an item

1. **One line: learned as a routine for rules 110 and 184, not for 30 and 90.** On a ring,
   trained on widths 6 to 12 with half the training batches run with whole steps, the
   machine run with whole steps gets every cell of every row right at widths 12, 24, 48 and
   96 for rule 110 on 2 of 3 seeds (and with 2 read pointers on the one seed tried) and for
   rule 184 on the one seed tried; rule 30 is not a routine on either of 2 seeds (0.75 and
   0.78 with whole steps) and rule 90 is not learned on either of 2 seeds (0.61, 0.67). Two
   read pointers are enough: the programs found with three use one or two and leave the
   rest idle.
2. **Several steps: not learned.** Rule 110 applied twice, one seed, 2,500 steps: 0.61 of
   cells at width 12 as trained and 0.46 with whole steps, where answering dead everywhere
   is 0.48; the write pointer never walks the output, and nothing is kept in scratch.
3. **One machine for many rules: not learned.** One seed, 2,000 steps: 0.61 of cells on
   rules never trained on at width 12 as trained, 0.50 with whole steps on held-out and
   trained rules alike (chance is 0.50); it never finds "seek the key, step to the value".
4. **Conway's game: not learned.** One seed, 2,000 steps, boards 4 by 4 to 6 by 6: 0.72 of
   cells at 6 by 6, 0.72 at 8 by 8 and 0.71 at 12 by 12, which is what answering dead
   everywhere scores (0.71, 0.72, 0.71); it does not get as far as walking the output.
5. **The trace: a loop, but not one vop and not three pointers.** After two ticks of setting
   up, one read pointer walks the ring one cell a tick and the write pointer follows one
   cell behind; the two cells just passed are carried in the pc, so the vop nearest the pc
   changes with the row (on 0.23 to 0.45 of ticks) among two or three vops.

## What was run

The world (the header of `vvm2/automata.py` has it in full):

- A cell is dead or live; each state is a fixed vector of 16 numbers from the world's code
  book, as in `vvm2/tape.py`. An output slot is read as the nearest of four book vectors.
- **Edges.** Both kinds were tried and the result is for wrapping edges. *Ring:* the input
  region is the row and the device's two steps make it a ring (`go`); the output is a line
  with one spare slot at each end that the world does not read. *Marks:* edges fixed at
  zero; the row sits between a left mark and a right mark, which count as dead and must be
  copied to the output.
- Trained on widths 6 to 12, batches of 64, AdamW, the schedule of `vvm2/tape.py`. Tested
  on 12, 24, 48, 96.
- Ticks: one an output slot plus 4 in everything below except the first two screens (two a
  slot plus 4). The task asked for a couple a cell; one was tried to halve the cost and
  that is where it learned. Two a slot was not tried again with the recipe that works.
- `whole=0.5`: half the training batches are run with whole steps (the core in `eval()`
  with `hard = True`, gradient on). On those batches the gradient reaches what is stored
  and the next pc, not the pointer moves. This is a training choice outside the core.
- Beside the machine: a convolutional net (one 3-wide layer of 16 channels, 276 weights),
  which has the neighbourhood built in; a GRU reading the row both ways and answering at
  each cell, once with the machine's weight count and once with its multiply-adds. The
  multiply-adds column counts fetch, execute and the seek match for the machine, and every
  weight once a position for the nets, on one board of width 12.

## 1. One line

### Rule 110 on a ring, whole=0.5, 3,000 steps, 3 read pointers (the result: 3 seeds)

| model | weights | multiply-adds | 12 | 24 | 48 | 96 |
|---|---|---|---|---|---|---|
| answering dead everywhere | | | 0.377 | 0.378 | 0.371 | 0.374 |
| vvm seed 0, as trained | 1,412,138 | 888,192 | 1.000 / 1.00 | 1.000 / 1.00 | 0.999 / 0.97 | 0.948 / 0.01 |
| vvm seed 0, whole steps | | | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| vvm seed 1, as trained | | | 1.000 / 1.00 | 0.979 / 0.64 | 0.870 / 0.00 | 0.750 / 0.00 |
| vvm seed 1, whole steps | | | 0.987 / 0.93 | 0.949 / 0.72 | 0.864 / 0.40 | 0.756 / 0.15 |
| vvm seed 2, as trained | | | 1.000 / 1.00 | 1.000 / 1.00 | 0.998 / 0.93 | 0.938 / 0.01 |
| vvm seed 2, whole steps | | | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| convolutional net (seed 0) | 276 | 3,312 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| GRU, the vvm's multiply-adds (seed 0) | 71,748 | 850,032 | 1.000 / 1.00 | 0.995 / 0.95 | 0.992 / 0.87 | 0.991 / 0.80 |
| GRU, the vvm's weights (seed 0) | 1,398,084 | 16,728,048 | 1.000 / 1.00 | 1.000 / 1.00 | 0.999 / 0.99 | 0.999 / 0.96 |

Two seeds of three are exact at eight times the trained width; the third is not a routine
(it falls with width either way it is run). The convolutional net is exact, as it should
be. The GRU is nearly exact and not quite: it also answers cell by cell, so this task does
not separate the machine from a recurrent net the way copy did (finding 33); at width 96
the machine has 300 of 300 rows right and the GRU 288 (same weights) or 240 (same
compute).

Bank use, whole steps, all test sizes: seed 0, 5 of 128 vops ever nearest, entropy 0.18,
top share 0.68; seed 1, 28 of 128, 0.42, 0.45; seed 2, 7 of 128, 0.26, 0.48. Little of the
bank, with an exact score, on the seeds that work.

### What else was tried on rule 110 (screens, seed 0, the machine only)

| edges | read pointers | ticks a slot | steps | whole | how run | 12 | 24 | 48 | 96 |
|---|---|---|---|---|---|---|---|---|---|
| marks | 2 | 2 | 1,500 | 0 | as trained | 0.769 / 0.06 | 0.658 / 0.00 | 0.632 / 0.00 | 0.628 / 0.00 |
| | | | | | whole steps | 0.365 / 0.00 | 0.386 / 0.00 | 0.375 / 0.00 | 0.375 / 0.00 |
| marks | 3 | 2 | 1,500 | 0 | as trained | 0.812 / 0.09 | 0.705 / 0.00 | 0.650 / 0.00 | 0.609 / 0.00 |
| | | | | | whole steps | 0.405 / 0.00 | 0.396 / 0.00 | 0.374 / 0.00 | 0.374 / 0.00 |
| marks | 3 | 1 | 4,000 | 0 | as trained | 0.997 / 0.97 | 0.833 / 0.07 | 0.678 / 0.00 | 0.594 / 0.00 |
| | | | | | whole steps | 0.413 / 0.00 | 0.391 / 0.00 | 0.382 / 0.00 | 0.382 / 0.00 |
| ring, one spare slot | 3 | 1 | 4,000 | 0 | as trained | 1.000 / 1.00 | 0.978 / 0.66 | 0.801 / 0.00 | 0.695 / 0.00 |
| | | | | | whole steps | 0.936 / 0.44 | 0.964 / 0.41 | 0.973 / 0.29 | 0.976 / 0.14 |
| ring | 3 | 1 | 3,000 | 0 | as trained | 1.000 / 1.00 | 0.998 / 0.96 | 0.968 / 0.31 | 0.902 / 0.00 |
| | | | | | whole steps | 0.759 / 0.06 | 0.767 / 0.00 | 0.770 / 0.00 | 0.763 / 0.00 |
| marks | 3 | 1 | 3,000 | 0.5 | as trained | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| | | | | | whole steps | 0.899 / 0.40 | 0.889 / 0.17 | 0.887 / 0.02 | 0.882 / 0.00 |
| ring | 2 | 1 | 3,000 | 0.5 | as trained | 1.000 / 1.00 | 1.000 / 0.99 | 0.972 / 0.25 | 0.852 / 0.00 |
| | | | | | whole steps | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |

Answering dead everywhere scores 0.37 to 0.38 in every column. The 2-pointer machine has
917,284 weights and 594,720 multiply-adds; bank use 6 of 128, entropy 0.24, top share 0.50.

What the screens say:

- Trained in the ordinary way the machine fits the trained widths and leans on blended
  pointers. A pointer spread over a few neighbouring cells reads their blend, which is a
  cheap view of a neighbourhood, so training has no reason to make the pointers exact.
  Copy did not have this way out (one cell is all a copy needs).
- Whole-step batches are what turned a blurred fit into a routine on the ring.
  The control on the same ring without them: 1.000 as trained at width 12 and 0.76 with
  whole steps at every width, against exact with them. One seed each way.
- With marks at the edges and whole-step batches the machine as trained is right on every
  row at every width to 96 (pointers sharp enough that nothing builds up), yet with whole
  steps it is 0.88 to 0.90: one of its read pointers still alternates between two places
  and the blend of the two is part of the answer. Not a routine by the rule of this repo,
  though it carries to eight times the width as trained. One seed.
- 2 or 3 read pointers: both learn it. See the trace: the third pointer is not used.

### The other rules (ring, 3 pointers, whole=0.5, 3,000 steps, the machine only)

Answering dead everywhere scores 0.50 for all three.

| rule | seed | how run | 12 | 24 | 48 | 96 |
|---|---|---|---|---|---|---|
| 184 | 0 | as trained | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 0.973 / 0.14 |
| 184 | 0 | whole steps | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| 30 | 0 | as trained | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 0.986 / 0.44 |
| 30 | 0 | whole steps | 0.766 / 0.06 | 0.754 / 0.00 | 0.748 / 0.00 | 0.753 / 0.00 |
| 30 | 1 | as trained | 0.865 / 0.31 | 0.775 / 0.05 | 0.701 / 0.00 | 0.611 / 0.00 |
| 30 | 1 | whole steps | 0.784 / 0.12 | 0.720 / 0.00 | 0.699 / 0.00 | 0.671 / 0.00 |
| 90 | 0 | as trained | 0.610 / 0.00 | 0.519 / 0.00 | 0.495 / 0.00 | 0.498 / 0.00 |
| 90 | 0 | whole steps | 0.484 / 0.00 | 0.497 / 0.00 | 0.495 / 0.00 | 0.497 / 0.00 |
| 90 | 1 | as trained | 0.672 / 0.02 | 0.560 / 0.00 | 0.524 / 0.00 | 0.504 / 0.00 |
| 90 | 1 | whole steps | 0.536 / 0.00 | 0.520 / 0.00 | 0.512 / 0.00 | 0.503 / 0.00 |

Rule 184 is learned like 110. Rule 30 on seed 0 finds a program that writes in step with
its reading pointer, with no lag, so the cell ahead can only come from a blended pointer;
it is right as trained and wrong with whole steps. On seed 1 it is not right either way.
Rule 90 (left exclusive-or right, the centre cell not used) does not get far below the
loss of a guess (0.69) on either seed: 0.64 and 0.49 at step 2,500, where 110 and 184 are
at about 0.01 or less by step 1,500. Copying the centre cell is no use for rule 90 and scores 0.63 for rule 110, so 110
and 184 have an easy first step that 90 lacks; that is a guess, not tested. No baselines
were run for these three rules (a 3-wide convolution can state all of them). Bank use,
seed 0: 184, 6 of 128, entropy 0.25; 30, 4 of 128, 0.19; 90, 6 of 128, 0.08, top share
0.91.

## 5. The traces

Run with whole steps on a ring of 8. Places: i3 the fourth input slot, o0 and o9 the spare
output slots, o1 to o8 the answers for cells 0 to 7. Each column is a tick: the vop nearest
the pc, the slot each pointer is on for that tick's load and store, and the move it makes.

Rule 110, 2 read pointers, seed 0 (exact at every width):

```
in:     ...##.#.
out:    #..#####.#          (first and last are the spare slots)
wanted: ...#####..
vop:      120   122    27   122   122   122    65   122    27    27    27   122   122   122
read 1:    i7    i0    i7    i3    i2    i3    i4    i0    i7    i6    i5    i3    i2    i3
  then:   in0  prev  seek  prev  seek  seek  seek  prev  prev  prev  seek  prev  seek  seek
read 2:    i0    i7    i6    i5    i4    i3    i2    i1    i0    i7    i6    i5    i4    i3
  then:   inN  prev  prev  prev  prev  prev  prev  prev  prev  prev  prev  prev  prev  prev
write:     o9    o9    o8    o7    o6    o5    o4    o3    o2    o1    o0    o0    o0    o0
  then:  outN  prev  prev  prev  prev  prev  prev  prev  prev  prev  prev  prev  prev  prev
```

Read 2 starts on cell 0, goes to the last cell and walks backwards round the ring, one cell
a tick, for ever. The write pointer waits one tick on the far spare slot and then walks
backwards one cell behind: when it is on the answer for cell 7 (o8), read 2 is on cell 6,
having been on cell 7 the tick before and on cell 0 the tick before that. Read 1 wanders
(seeks that land anywhere) and the answer is still exact, so the two cells already passed
can only be reaching the vop through the pc. After the row is done the write pointer
stands on the near spare slot. The vop nearest the pc is 122, 27 or 65 according to the
cells just passed: over 200 rows, 3.7 different vops a run, and on 0.45 of ticks a row's
vop is not the commonest one for that tick.

Rule 110, 3 read pointers, seed 0 (exact): the same program. Read 3 walks backwards, the
write pointer one behind, reads 1 and 2 seek on every tick and are not needed. One vop (95)
on 0.66 of ticks; the pick changes with the row on 0.23 of ticks.

```
vop:       48    71    95    95    95    95    95    19    95    95    95    95    95    95
read 3:    i0    i0    i7    i6    i5    i4    i3    i2    i1    i0    i7    i6    i5    i4
write:     o9    o9    o9    o8    o7    o6    o5    o4    o3    o2    o1    o0    o0    o0
```

Rule 110, 3 read pointers, seed 2 (exact): two pointers a cell apart walk backwards and the
write pointer sits level with the trailing one, so one cell comes through the pc and two
through pointers. Read 1 parks on the spare output slot.

```
vop:       77    99   103    23    23    23    72   104    23    72    23    23    23    23
read 2:    i7    i7    i6    i5    i4    i3    i2    i1    i0    i7    i6    i5    i4    i3
read 3:    i7    i0    i7    i6    i5    i4    i3    i2    i1    i0    i7    i6    i5    i4
write:     o9    o9    o8    o7    o6    o5    o4    o3    o2    o1    o0    o0    o0    o0
```

Rule 184, seed 0 (exact): the first program again (one pointer walking backwards, the write
pointer one behind, vops 39, 49, 50 by the cells passed).

So: it is a loop and the pointers step, and it is exact at any width tried. It is not "one
vop on every tick" as copy was, because the routine keeps its window of cells in the pc and
the pc is also what picks the vop. That the pc carries the cells is read off the traces
(nothing else could hold them); the pc was not probed.

## 2, 3, 4. Several steps, a rule table, Conway's game (one seed each, short runs)

All three use the recipe that worked for item 1 (whole=0.5) and none is learned. The large
GRU was left out to save time, so the nets beside these are the local one and the GRU with
the machine's multiply-adds.

**2. Rule 110 applied twice** (ring, 3 pointers, 2,500 steps, scratch as long as the row
plus four, twice the ticks: 2 times (slots + 4)).

| model | weights | multiply-adds | 12 | 24 | 48 | 96 |
|---|---|---|---|---|---|---|
| answering dead everywhere | | | 0.479 | 0.457 | 0.464 | 0.469 |
| vvm, as trained | 1,412,138 | 1,744,128 | 0.606 / 0.01 | 0.539 / 0.00 | 0.526 / 0.00 | 0.537 / 0.00 |
| vvm, whole steps | | | 0.459 / 0.01 | 0.471 / 0.00 | 0.467 / 0.00 | 0.468 / 0.00 |
| convolutional net, two layers | 1,060 | 12,720 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| GRU, the vvm's multiply-adds | 143,428 | 1,705,584 | 0.841 / 0.17 | 0.788 / 0.02 | 0.772 / 0.00 | 0.764 / 0.00 |

The loss sits at 0.63 to 0.70 from step 1,000 on (a guess is 0.69). In the trace the write
pointer seeks between scratch and input and never reaches the output; the read pointers
seek or stay. Bank use: 18 of 128 ever nearest, entropy 0.13, top share 0.81: one vop on
most ticks whatever the row, with a poor score, which is the locked-in case. On a ring the
second pass could not be the same loop in any case (see "What is missing"); the machine
could have learned the two-step rule as one wider window in the pc, and did not in this
budget. The GRU has not finished learning it at 2,500 steps either.

**3. The rule given as a table** (marks at the edges, 4 read pointers, 3 ticks a slot plus
6, 2,000 steps; 192 rules trained on, 64 held out with 110, 30, 90 and 184 among them; the
8 key and value pairs in a fresh order every board). Scores are on the held-out rules.

| model | weights | multiply-adds | 12 | 24 | 48 |
|---|---|---|---|---|---|
| answering dead everywhere | | | 0.504 | 0.508 | 0.496 |
| vvm, as trained | 2,013,488 | 3,379,968 | 0.608 / 0.03 | 0.608 / 0.02 | 0.345 / 0.00 |
| vvm, whole steps | | | 0.502 / 0.01 | 0.513 / 0.00 | 0.493 / 0.00 |
| vvm, whole steps, on the trained rules | | | 0.501 / 0.02 | 0.508 / 0.01 | 0.523 / 0.01 |
| a net given the table and one neighbourhood | 1,860 | 26,040 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| GRU, the vvm's multiply-adds | 113,668 | 3,375,480 | 0.798 / 0.21 | 0.776 / 0.07 | 0.753 / 0.07 |

The loss is still falling slowly at the end (0.56 at step 1,000, 0.49 at 2,000). In the
trace the write pointer parks on the last output slot and three read pointers park on the
first; one read pointer seeks between two table slots. Nothing like "seek the key, step to
the value, write it, step on". Bank use: 29 of 128, entropy 0.23, top share 0.74. Lookup
by key alone took 16,000 steps to learn (finding 36), so 2,000 steps here says little
about whether it can be learned; it says it is not quick.

**4. Conway's game** (3 read pointers, 4 steps on the grid, 1 tick an output slot plus 6,
2,000 steps, boards 4 by 4 to 6 by 6, between 0.15 and 0.60 of cells live). Output layout:
the board's rows in reading order, each with its left and right mark, so a board of n by n
is n times (n + 2) output slots; the device's "right" goes in reading order, on from the
end of a row to the start of the next.

| model | weights | multiply-adds (6 by 6) | 6 by 6 | 8 by 8 | 12 by 12 |
|---|---|---|---|---|---|
| answering dead everywhere | | | 0.705 | 0.717 | 0.709 |
| vvm, as trained | 1,495,082 | 3,100,032 | 0.719 / 0.07 | 0.717 / 0.01 | 0.711 / 0.00 |
| vvm, whole steps | | | 0.711 / 0.03 | 0.719 / 0.01 | 0.715 / 0.00 |
| convolutional net, 3 by 3 | 932 | 59,648 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| GRU, the vvm's multiply-adds | 51,204 | 3,227,904 | 0.804 / 0.04 | 0.741 / 0.01 | 0.704 / 0.00 |

How far it gets: nowhere past "mostly dead". The loss goes from 1.31 to 0.93 and stops;
the write pointer seeks about the input region (where it cannot write) and comes back to
the first output slot, and never walks the output. Bank use: 17 of 128, entropy 0.37, top
share 0.43. The whole boards it gets right are most likely boards whose answer is all dead
(not checked). The GRU does what was expected of it: above the dead line at the trained
size and back on it at 12 by 12, where the cell above is a different number of slots away.
One tick a slot is enough for a program that keeps three pointers in a column and two
column sums in the pc; 3 ticks a slot (the default in the script) was not run because of
its cost.

## What is missing in the machine (nothing was changed; these are what the tasks ran into)

- **The device's steps cover the input region only.** Scratch and output are lines that
  stop at their ends. A ring kept in scratch has no wrap step, so "apply the rule again to
  the row in scratch" cannot be the same loop on a ring; and a grid kept in scratch has no
  down or up at all. A second pass over the machine's own writing needs either steps a
  device can define on scratch, or a device that takes writes.
- **Nothing tells a task how exact the pointers are in a way a loss can use.** `sharp()`
  and the trace are cut off from the gradient. The whole-step batches used here are the way
  round it from outside.
- **A pass cannot end itself.** On a ring nothing marks the last cell and the machine cannot
  halt, so a write pointer that has finished goes on writing. The world gave it a spare
  slot at each end to stand on. With one spare slot the program that wrote back to front
  ended parked on an answer.

## What is suspect

- **Seeds.** Three seeds for rule 110 with 3 pointers and whole=0.5, of which one failed.
  Rules 30 and 90 have two seeds. Everything else is one seed: 2 pointers, rule 184, marks
  at the edges, the control, and items 2 to 4.
- **Items 2 to 4 are short runs.** 2,000 to 2,500 steps, where lookup alone needed 16,000
  (finding 36). They show these are not learned quickly, not that they cannot be.
- **Whole-step batches are new here** and were tested on this task only. The control
  without them is one seed. Whether they help or hurt copy, reverse and lookup is not known.
- **Ticks.** The result is at one tick an output slot. Two ticks a slot were only run in the
  first two screens (1,500 steps, marks, no whole-step batches), which failed for several
  possible reasons at once.
- **The world helps.** The ring comes from the device's steps and the spare output slots
  are there so a finished write pointer does no harm. Neither tells the machine the rule,
  but with marks at the edges, where the ends are special, no run was exact with whole
  steps.
- **The carried cells are inferred.** "The pc holds the last two cells" is read off traces
  in which no pointer is on those cells and the answer is exact. The pc was not probed.
- **The GRU is nearly as good.** A GRU that answers cell by cell has the alignment built in;
  at the machine's weight count it has 0.999 of cells at width 96. The machine's edge is
  "exact", 300 rows of 300 against 288, on two seeds of three. A GRU that reads the row and
  then writes it out (the kind beside copy in `vvm2/tape.py`) was not run.
- **Baselines** were run on seed 0 of rule 110 only (and the smaller two on items 2 to 4).
- **Mixed conditions.** The first four screens ran 4 threads on the core before its tick
  was made faster (same numbers to 5e-6, per the note with that change); later runs are 1
  thread on the faster core. Step counts are 1,500, 4,000 and 3,000 as marked. The screen
  "ring, one spare slot" cannot be rerun from the script as it now is (two spare slots).
- **Multiply-adds** are counted by formula on a board of width 12, not measured. On this
  shared CPU a 3,000-step run of the 3-pointer machine took 12 to 35 minutes.
- Each test is 300 boards a size. Whole boards right at width 96 is 300 rows of 96 cells.

## What was not done

- Rules 30 and 90 on more seeds or more steps; baselines for rules 30, 90, 184.
- Two ticks a slot with whole-step batches; k given as an input slot; k = 4.
- More than one seed, or a longer run, for items 2, 3 and 4; the large GRU beside them;
  Conway's game at 3 ticks a slot; the rule applied k times with marks at the edges.
- Probing the pc; a bank smaller than 128 (5 to 7 vops are used).
- Whole-step batches on copy, reverse, lookup or the lights game.
- A hand-written program in the frozen core, to show the three-pointer loop can be stated.

## The exact script lines

```
scripts/setup.sh
scripts/automata.sh --quick

# the result: rule 110 on a ring, 3 seeds (seed 0 with the nets beside it)
scripts/automata.sh line 3000 reads=3 tpc=1 edge=ring whole=0.5 seed=0
scripts/automata.sh line 3000 reads=3 tpc=1 edge=ring whole=0.5 seed=1 base=0
scripts/automata.sh line 3000 reads=3 tpc=1 edge=ring whole=0.5 seed=2 base=0

# screens, seed 0
scripts/automata.sh line 1500 reads=2 base=0                         # marks, 2 ticks a slot
scripts/automata.sh line 1500 reads=3 base=0
scripts/automata.sh line 4000 reads=3 tpc=1 base=0                   # marks, 1 tick a slot
scripts/automata.sh line 3000 reads=3 tpc=1 edge=ring base=0         # the control
scripts/automata.sh line 3000 reads=3 tpc=1 whole=0.5 seed=0 base=0  # marks, whole-step batches
scripts/automata.sh line 3000 reads=2 tpc=1 edge=ring whole=0.5 base=0
scripts/automata.sh line 3000 rule=30  reads=3 tpc=1 edge=ring whole=0.5 base=0
scripts/automata.sh line 3000 rule=90  reads=3 tpc=1 edge=ring whole=0.5 base=0
scripts/automata.sh line 3000 rule=184 reads=3 tpc=1 edge=ring whole=0.5 base=0
scripts/automata.sh line 3000 rule=30  reads=3 tpc=1 edge=ring whole=0.5 seed=1 base=0
scripts/automata.sh line 3000 rule=90  reads=3 tpc=1 edge=ring whole=0.5 seed=1 base=0

# items 2, 3, 4 (one seed, short)
scripts/automata.sh line 2500 k=2 reads=3 tpc=1 edge=ring whole=0.5 base=2
scripts/automata.sh rules 2000 reads=4 whole=0.5 base=2
scripts/automata.sh life 2000 tpc=1 whole=0.5 base=2
```

Each line is one job on one thread at nice 19 and prints its table, bank use and trace.

## Proposed edits (not made: docs/findings.md and docs/ideas.md are the owner's)

For `docs/findings.md`, after 37:

38. **A one-line cellular automaton is learned as a routine that runs at eight times the
    trained width, when half the training batches are run with whole steps.** [likely: 3
    seeds for rule 110, 2 exact and 1 not; one seed each for 2 pointers and rule 184;
    `scripts/automata.sh line 3000 reads=3 tpc=1 edge=ring whole=0.5 seed=N`] Rule 110 on a
    ring, trained on widths 6 to 12: run with whole steps every cell of every row is right
    at 12, 24, 48 and 96 on seeds 0 and 2, and with 2 read pointers, and for rule 184; seed
    1 is 0.99 falling to 0.76. A GRU answering cell by cell has 0.999 of cells and 0.96 of
    rows at width 96 with the same weights, 0.991 and 0.80 with the same compute; a 276
    weight convolution is exact. Rule 30 is right as trained to width 96 (0.986) and 0.75
    with whole steps; rule 90 is not learned (0.61). 5 to 7 of 128 vops are used.
39. **Trained the ordinary way, the machine fits a neighbourhood rule with blurred pointers
    and the fit does not survive whole steps.** [hint: one seed a setting] Rule 110, 4,000
    steps: 0.997 of cells as trained at width 12 and 0.41 with whole steps (marks at the
    edges); 1.000 and 0.94 (ring); 3,000 steps on the ring, 1.000 and 0.76. A pointer spread over neighbouring cells
    reads their blend, which is a cheap view of a neighbourhood; copy had no such way out.
    Whole-step batches (the core in eval with `hard = True`, gradient on, so the loss sees
    the machine as it will be run) are what removed it.
40. **The routine it finds keeps its window in the pc, so it is a loop but not one vop.**
    [likely: traces of the four exact runs] One read pointer walks the ring a cell a tick,
    the write pointer one cell behind, and the two cells already passed ride in the pc; the
    vop nearest the pc is one of two or three according to those cells (it changes with the
    row on 0.23 to 0.45 of ticks). Given three read pointers it uses one or two. Not
    probed: read off traces in which no pointer is on the carried cells.
41. **The rule applied twice, the rule given as a table, and Conway's game are not learned
    in short runs.** [hint: one seed each, 2,000 to 2,500 steps, `scripts/automata.sh line
    2500 k=2 ...`, `... rules 2000 ...`, `... life 2000 ...`] Twice: 0.61 of cells at width
    12 (dead everywhere 0.48). Table: 0.61 on held-out rules (chance 0.50), 0.50 with whole
    steps. Conway: 0.72 at every size, the score of answering dead everywhere. In none does
    the write pointer walk the output. The steps a device defines reach the input region
    only, so a second pass over a row or grid the machine has written has no ring or grid
    to walk on.

For `docs/ideas.md`, after 62:

63. **Whole-step batches as a standing part of training** (from the automata runs): some
    share of batches run with `hard = True`, so nothing can be learned that only works
    blurred. Try it on copy, reverse, lookup and the lights game (finding 34 is the same
    fault: a fit that depends on the blur). A straight-through move would let those batches
    teach the pointers too; it needs a change inside the tick.
64. **Steps a device can define on scratch**, or a device that takes writes: a second pass
    over the machine's own output (a rule applied k times, a grid kept between steps) has
    only a line to walk on now.
65. **A pointer-exactness number the loss can see**, as a cheaper way than whole-step
    batches to ask for exact pointers.
66. **The pc doing two jobs.** In the automaton routine the pc is both the window of cells
    and the choice of vop, so the "program" is a small table of vops indexed by data. A
    register slot beside the pc (read every tick, not used for the fetch) would let data be
    carried without changing what runs; whether that is wanted is a design question.
67. **An easy first step matters.** Rules 110 and 184 are mostly "copy the centre cell" and
    were learned; rule 90 ignores the centre cell and was not. A curriculum of data (idea
    58) for rules with no easy first step: rule 204 (copy), then the target rule.


## How the predictions did

- Item 1, three pointers learn a loop on 2 of 3 seeds: right only for the ring with
  whole-step batches, which is not the setting it was written for (marks, ordinary
  training: 0.41 with whole steps). "Right at 12 and 24 as trained, falling at 48 and 96":
  right in kind (0.999 and 0.948 on seed 0).
- Two pointers do worse: wrong. Two pointers are exact.
- Rules 30, 90, 184 go the way of 110, and later "at least two of the three exact": wrong
  both times. Only 184.
- The ring is learned: right. The convolution 1.000 and the GRU 0.97 or more at 96: right.
- Second predictions: the ring with whole-step batches 0.99 or more with whole steps:
  right on seeds 0 and 2, wrong on seed 1. Marks between 0.41 and 0.90: right, just
  (0.899 at width 12).
- Third: the control under 0.99 with whole steps: right (0.76).
- Fourth: rule 90 seed 1 under 0.75: right (0.67). Rule 30 seed 1 at 0.99 as trained:
  wrong (0.87); not exact with whole steps: right.
- Items 2, 3, 4 not learned: right, at fewer steps than the predictions named (2,500,
  2,000, 2,000 for 6,000). "No whole board right at 8 by 8": wrong by about 3 boards of
  300. The GRU on Conway: right.
- The trace, "one vop on every tick, every pointer stepping, under 0.05 of ticks differ":
  wrong. One or two pointers step, the rest idle, and the vop changes with the row on 0.23
  to 0.45 of ticks.

## What had been seen before the predictions were written

Only `scripts/automata.sh --quick`: every task for 20 training steps with a bank of 16 vops,
to see that the code runs. Nothing learns in 20 steps and no number from it is used.

## Predictions (written before any scoring run)

Scores are share of cells right / share of whole boards right. "Whole steps" is the machine
run with `core.hard = True`. A screen is 1 or 2 seeds; a result is 3.

**1. One line.** Rule 110, fixed-zero edges shown as marks, trained on widths 6 to 12, 3,000
steps, tested on 12, 24, 48, 96.

- The machine with 3 read pointers learns it as a loop: with whole steps 0.99 or more of
  cells at width 12 and the same at width 96, on at least 2 of 3 seeds (60% sure). As
  trained (blended steps) it is right at 12 and 24 and falls at 48 and 96, as copy did.
- With 2 read pointers it does worse at the same budget: under 0.95 of cells at width 12,
  and worse again run with whole steps (65% sure). It has to carry the left neighbour in
  the pc or take two ticks a cell, which is a harder program to find.
- Rules 30, 90 and 184 go the same way as 110 with 3 pointers. No rule fails where another
  succeeds (55% sure; 30 and 90 need an exclusive-or and may be slower).
- Wrapping edges (the device a ring, a spare output slot): also learned with 3 pointers.
- The convolutional net: 1.000 at every width. The GRU read both ways, either size: 1.00 at
  12 and 0.97 or more at 96, since it also answers cell by cell.

**2. Several steps** (the rule applied twice). Not learned as two passes: under 0.85 of
cells at width 12 and under 0.10 of whole rows, at 6,000 steps (75% sure). If it scores
well at trained widths, it will not carry to 48 and 96.

**3. One machine for many rules.** Not learned in 6,000 steps: under 0.75 of cells on rules
it never trained on (answering at random is 0.5) (70% sure). The GRU does better than the
machine on trained rules at width 12 and loses some on held-out rules and longer rows.

**4. Conway's game.** Not learned: cells right within 0.10 of "answering dead everywhere"
at every size, no whole board right at 8 by 8 (80% sure). The convolutional net: 1.00. The
GRU: clearly above the dead line at 6 by 6 and back near it at 12 by 12, because the cell
above is a different number of slots away on a larger board.

**5. The trace.** Where item 1 is learned with 3 pointers: one to three ticks of setting
up, then one vop on every tick with every pointer taking the step "next", and the picked
vop the same whatever the row (under 0.05 of ticks differ between rows).

## Second predictions (written after the first four screens, before the next runs)

Seen so far, one seed each, rule 110: at 1,500 steps and 2 ticks a slot neither 2 nor 3
pointers learn anything that survives whole steps (0.77 and 0.81 of cells as trained, 0.37
and 0.41 with whole steps, where answering dead everywhere is 0.37). At 4,000 steps and 1
tick a slot with 3 pointers: marks at the edges 0.997 as trained at width 12 and 0.41 with
whole steps; the ring 1.000 as trained and 0.94 to 0.98 with whole steps at every width.
So the trained machine leans on blended pointers, which read a blur of neighbouring cells.

Two changes follow, both outside the core. The ring's output line gets a spare slot at each
end (the seed-0 ring program wrote back to front and ended parked on a cell it had
answered). And `whole=0.5`: half the training batches are run with whole steps, so the
loss also sees the machine as it will be run; the gradient on those batches reaches what is
stored and the pc but not the pointer moves.

- Ring, whole=0.5, 4,000 steps, 1 tick a slot: with whole steps 0.99 or more of cells at
  every width (50% sure).
- Marks at the edges, whole=0.5: with whole steps better than 0.41 and under 0.90 (55%
  sure): the pointer paths it finds there are not near a loop, and whole-step batches
  cannot move pointers.

## Third predictions (after the three-seed ring runs, before the runs on other rules)

Seen: ring, whole=0.5, 3,000 steps (cut from 4,000 to save time), 3 pointers: seeds 0 and 2
exact at every width with whole steps, seed 1 not (0.99 falling to 0.76). The two exact
programs step only one or two read pointers and carry the other cells in the pc.

- 2 read pointers, same recipe, seed 0: learned, exact with whole steps at every width (60%
  sure), since the 3-pointer programs did not use the third pointer.
- Rules 30, 90, 184, same recipe, seed 0: at least two of the three exact with whole steps
  at every width (55% sure).
- Without whole-step batches on the same ring layout (the control): 1.00 as trained at
  width 12, under 0.99 with whole steps (65% sure).
- Items 2, 3, 4: as first predicted, now at 3,000 steps or fewer.

## Fourth predictions (before a second seed of rules 90 and 30)

Seen: seed 0 of rule 90 not learned (0.61), of rule 30 right as trained and 0.75 with whole
steps, of rule 184 exact; the control without whole-step batches 0.76 with whole steps.

- Rule 90, seed 1: not learned again, under 0.75 of cells at width 12 as trained (60% sure).
- Rule 30, seed 1: 0.99 or more as trained at width 12; not exact with whole steps at width
  96 (55% sure).
