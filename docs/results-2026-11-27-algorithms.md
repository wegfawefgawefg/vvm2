# Small algorithms, one conditional, and the lights game at long holds (2026-11-27)

Code: `vvm2/algos.py` (the ladder and the conditional), `vvm2/tape.py` (the lights game, with
two new options: `clock=1` and `threads=N`). Script: `scripts/algos.sh`. The core is not
touched.

## Predictions (written and committed before any scoring run)

Every task: trained on lengths 2 to 8, 3,000 steps, tested at 8, 16, 32 and 64. "Carries"
means run with whole steps it is right on nearly every whole answer at 64.

The ladder, in the order I expect from easy to hard:

| # | task | learned at trained lengths | carries, run with whole steps | the GRU of the same weights |
|---|---|---|---|---|
| 1 | shift | yes | yes | right at 8, fails past it |
| 2 | double (each item twice) | yes | yes, 70% sure | right at 8, fails past it |
| 3 | keep (filter) | yes | yes, 60% sure | right at 8, fails past it |
| 4 | count (tally) | yes | even odds | right at 8, partly past it |
| 5 | twice (list, then list again) | mostly (items above 0.9) | no (items under 0.6 at 64) | right at 8, fails past it |
| 6 | parity | no (under 0.7 at 8; chance 0.5) | no | better than the machine at 8; may partly carry |
| 7 | compare | partly (0.75 to 0.9; chance 0.5) | no | above 0.95 at 8, near 0.6 at 64 |
| 8 | add, binary | partly (items near 0.8) | no | above 0.95 at 8, fails past it |
| 9 | sort (4 symbols) | partly (items near 0.8) | no | right at 8, partly past it |
| 10 | brackets | partly (near 0.75) | no | better than the machine |
| 11 | add, decimal | no (items under 0.5) | no | partly |

Why: shift, double, keep and count can each be done by one vop that runs every tick, since
a vop's pointer steps and its stored value are a linear function of the item it loaded: no
change of pc is needed for "write this one or not" or "step the write pointer or not". Twice
needs the pc to change once at a time that depends on the data (the end mark). Parity needs
the pc to change at every item. Compare needs "are these two vectors the same", which one
matrix cannot say. So I expect the ladder to stop at twice and parity, and by the rule (stop
after two failures in a row) rungs 7 to 11 may not be climbed.

B, one conditional (`cond`: the signal slot says copy or reverse, and stays lit): learned at
trained lengths; carries to 64 with whole steps on at least one seed of three; 60% that the
vops run differ with the signal on most ticks, 40% that it keeps one read pointer on the
signal slot and uses the same vops for both. `condflash` (the signal goes dark after the
first tick, so the pc or a scratch slot must hold it): even odds it is learned at trained
lengths, one in three that it carries.

C, the lights game:
- one tick a light, 16,000 steps: right at 6 lights as trained; under 0.6 with whole steps;
  pointers under 0.8 exact; no clean "store one light a scratch slot and step" (70% sure).
- lights held 2, 4, 8, 16 ticks, 3,000 steps, at 6 lights: the machine about 0.9, 0.6, then
  0.25 to 0.4 at 8 and 16 (chance 0.25); the GRU 1.0, 1.0, 0.8, 0.5. So I expect long holds
  to hurt the machine more than the GRU, against the owner's guess: the machine has to
  count ticks with a ring of vops, and the GRU can count in its state.
- with the clock slot: the machine better at 4 and 8 ticks by 0.1 to 0.3; the GRU a little
  better. Neither carries to 16 lights.
