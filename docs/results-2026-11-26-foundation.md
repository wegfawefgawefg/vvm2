# Foundation and first numbers (plan 0a to 0d, a first look at 0e)

2026-11-26. Status: **predictions only**. This file was committed with the predictions
below before any scoring run was started; the results sections are filled in afterwards
and the predictions are not edited.

## What was built

- `vvm2/bank.py`: the program bank. Product keys, exact top-k, softmax over the k scores,
  sparse value gradients, the pick and query noise, the use counters.
- `vvm2/machine.py`: the machine (registers, input write, bank, ALU, ticks, readout), its
  parts switchable. `vvm2/memory.py`: the data memory, version 0.
- `vvm2/baselines.py`: MLP, GRU, transformer, and sizing to a target.
- `vvm2/tasks.py`: the task generators. `vvm2/train.py`: one run, one results row.
  `vvm2/table.py`: the tables. `vvm2/probe.py`: the look inside for 0d.
  `vvm2/check.py`: checks of the lookup, the sparse gradient and the accounting.
- `scripts/`: one script per experiment (recipe in its header), `smoke.sh`, `pilot.sh`.

## What had been seen before the predictions were written

Pilots on settings that are in no results table (`scripts/pilot.sh`), one seed each, and
the smoke test. They are the only runs read before the predictions:

- Storage, 16384 items in a bank of 1024 entries, 3000 steps: the machine and the three
  parameter-matched baselines all reached 1.000 with lr 3e-3 (the machine with bank lr
  1e-2 or 3e-2); all were slower with lr 1e-3. One tick stored 0.53 where two ticks stored
  0.75 (at lr 1e-3). Bank use was above 0.99 with entropy 0.97. So: the learning rates in
  the scripts, and an expectation that the bank is not collapsed at these sizes.
- Modular addition with p = 59, 4000 steps: the parameter-matched MLP and transformer
  reached 0.98 and 1.00 on the held-out pairs only with lr 3e-3 and weight decay 1, and
  stayed at chance otherwise. The machine stayed at chance (0.004 to 0.011) at every
  setting, with full training accuracy. So: weight decay 1 in 0c and 0d, and the
  prediction below that the machine does not generalise here.
- Smoke-sized pointer chasing (16 and 32 items, a few hundred steps): the machine fitted
  the training pairs and got none of the held-out pairs.

## Predictions (written before any scoring run)

Chance is 1/64 for storage, 1/97 for modular addition, 1/2 for parity, 1/256 for chasing.

**0b storage.** Reference machine: bank 4096 entries, 2 ticks, about 298k parameters and
56k multiply-adds.

| line | 4096 items | 16384 items | 65536 items |
|---|---|---|---|
| machine | 1.00 | 0.99 or more | 0.90 to 1.00 |
| machine T=1 | 1.00 | about 0.9 | 0.4 to 0.7 |
| machine N=1024 | 1.00 | 0.95 or more | 0.25 to 0.5 |
| machine N=65536 | - | 1.00 | 1.00 |
| machine no bank | 0.8 to 1.0 | 0.2 to 0.4 | under 0.1 |
| mlp params | 1.00 | 1.00 | 0.85 to 1.0 |
| mlp madds | 1.00 | 0.8 to 1.0 | 0.15 to 0.3 |
| gru params | 1.00 | 1.00 | 0.7 to 1.0 |
| gru madds | 0.5 to 0.9 | under 0.2 | under 0.05 |
| transformer params | 1.00 | 1.00 | 0.8 to 1.0 |
| transformer madds | 0.4 to 0.8 | under 0.15 | under 0.05 |

In a sentence: at equal compute the machine stores several times more pairs than any
baseline and the gap widens with the bank (N=65536 stores all 65536 pairs for about 81k
multiply-adds); at equal parameters it is level with the MLP, no better. Bank use: above
0.95 of entries with entropy above 0.9 for the 4096-entry bank at every size; for the
65536-entry bank 0.5 to 0.9 of entries at 65536 items and 0.3 to 0.6 at 16384.

**0c modular addition.** Machine (2 or 4 ticks): full training accuracy, held-out under
0.05 on every seed. With decay of the read rows: still under 0.2 (I give it 30% to reach
0.9). Machine with no bank (the tied ALU alone): 0.9 or more on at least 2 of 3 seeds (60%
sure). MLP and transformer at equal parameters: 0.95 or more. At equal compute, and the
GRU at either size: 0.9 or more on at least 2 of 3 seeds, less sure (55%).

**0c parity, 12 bits.** With weight decay 0: MLP, transformer and the machine (whole
input, with or without the bank) at 0.45 to 0.6 held-out; the parameter-matched GRU at
0.95 or more on at least 2 of 3 seeds; the machine fed one bit a tick at 0.95 or more on
at least one seed but not on all. With weight decay 1: not piloted; the MLP has perhaps a
40% chance of generalising, the GRU may be hurt.

**0d pointer chasing.** With the task as set (loss on the answer only): every line, at
both weight decays, under 0.05 on the held-out pairs. That is the machine at 1, 2, 4 and 6
ticks, with the input held, with the small bank, with read-row decay, without the bank,
without the ALU, and every baseline including the GRU with extra steps. The walk does not
form on its own: the look inside the 4-tick machine shows no diagonal; the state reads out
as the final answer (or as nothing) from the first ticks on, not as f applied tick-many
times. With the readout supervised after every tick: held-out 0.8 or more at 4 ticks (60%
sure; if not, under 0.2), a little lower at 6 ticks, and lower by 0.2 or more when the
input is held on every tick (55% sure).

**0e collapse, first look.** With no remedy the bank is not collapsed but partly unused:
storage (65536 entries, 16384 items) 0.4 to 0.7 of entries used, entropy 0.85 to 0.95, top
entry under 0.001 of the weight; chasing (4096 entries) 0.5 to 0.8 used. Remedies, as the
change in the share of entries used in a clean pass: Gumbel noise on the scores the most
(up 0.15 or more); query noise next (about 0.1); picking 8 of the top 32 at random raises
the share that ever got a gradient to about 1.0 but the clean-pass share by 0.05 to 0.15;
state noise under 0.05 and it costs storage accuracy. No query normalisation: use down by
0.2 or more (60% sure). Batch norm on the query: the highest use of the three norms.
Storage accuracy 0.99 or more for every line except state noise.
