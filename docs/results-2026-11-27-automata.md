# Cellular automata, learned on small boards and run on larger ones

2026-11-27. Code: `vvm2/automata.py`, `scripts/automata.sh`. The core is not changed.

Status: predictions only. Nothing below has been scored yet.

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
