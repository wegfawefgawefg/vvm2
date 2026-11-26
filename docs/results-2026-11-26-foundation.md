# Foundation and first numbers (plan 0a to 0d, a first look at 0e)

2026-11-26. Status: **stopped early at the owner's request.** The foundation (0a) is built
and `scripts/smoke.sh` passes. Of the scoring runs only 0d (pointer chasing) ran, and not
quite to the end. 0b, 0c and 0e have scripts and predictions and no scoring runs. The
predictions below were committed (33aafc7) before any scoring run and are not edited.

In a sentence each:

- **0a**: built; see "What was built" and "Where this differs from docs/design.md".
- **0b, 0c, 0e**: not run. The only numbers are the one-seed pilots under "What had been
  seen before the predictions were written".
- **0d**: the walk does not form on its own. Every vvm line and every baseline fits
  the training pairs and stays near chance on the held-out pairs. When the readout is
  supervised after every tick the vvm does walk on the pairs it trained on, and a
  third of the held-out pairs follow (0.37 against chance 0.004), fewer the longer the
  walk.

## What was built

- `vvm2/bank.py`: the program bank. Product keys, exact top-k, softmax over the k scores,
  sparse value gradients, the pick and query noise, the use counters.
- `vvm2/vvm.py`: the vvm (registers, input write, bank, ALU, ticks, readout), its
  parts switchable. `vvm2/memory.py`: the data memory, version 0.
- `vvm2/baselines.py`: MLP, GRU, transformer, and sizing to a target.
- `vvm2/tasks.py`: the task generators. `vvm2/train.py`: one run, one results row.
  `vvm2/table.py`: the tables. `vvm2/probe.py`: the look inside for 0d.
  `vvm2/check.py`: checks of the lookup, the sparse gradient and the accounting.
- `scripts/`: one script per experiment (recipe in its header), `smoke.sh`, `pilot.sh`.

## What had been seen before the predictions were written

Pilots on settings that are in no results table (`scripts/pilot.sh`), one seed each, and
the smoke test. They are the only runs read before the predictions:

- Storage, 16384 items in a bank of 1024 entries, 3000 steps: the vvm and the three
  parameter-matched baselines all reached 1.000 with lr 3e-3 (the vvm with bank lr
  1e-2 or 3e-2); all were slower with lr 1e-3. One tick stored 0.53 where two ticks stored
  0.75 (at lr 1e-3). Bank use was above 0.99 with entropy 0.97. So: the learning rates in
  the scripts, and an expectation that the bank is not collapsed at these sizes.
- Modular addition with p = 59, 4000 steps: the parameter-matched MLP and transformer
  reached 0.98 and 1.00 on the held-out pairs only with lr 3e-3 and weight decay 1, and
  stayed at chance otherwise. The vvm stayed at chance (0.004 to 0.011) at every
  setting, with full training accuracy. So: weight decay 1 in 0c and 0d, and the
  prediction below that the vvm does not generalise here.
- Smoke-sized pointer chasing (16 and 32 items, a few hundred steps): the vvm fitted
  the training pairs and got none of the held-out pairs.

## Predictions (written before any scoring run)

Chance is 1/64 for storage, 1/97 for modular addition, 1/2 for parity, 1/256 for chasing.

**0b storage.** Reference vvm: bank 4096 entries, 2 ticks, about 298k parameters and
56k multiply-adds.

| line | 4096 items | 16384 items | 65536 items |
|---|---|---|---|
| vvm | 1.00 | 0.99 or more | 0.90 to 1.00 |
| vvm T=1 | 1.00 | about 0.9 | 0.4 to 0.7 |
| vvm N=1024 | 1.00 | 0.95 or more | 0.25 to 0.5 |
| vvm N=65536 | - | 1.00 | 1.00 |
| vvm no bank | 0.8 to 1.0 | 0.2 to 0.4 | under 0.1 |
| mlp params | 1.00 | 1.00 | 0.85 to 1.0 |
| mlp madds | 1.00 | 0.8 to 1.0 | 0.15 to 0.3 |
| gru params | 1.00 | 1.00 | 0.7 to 1.0 |
| gru madds | 0.5 to 0.9 | under 0.2 | under 0.05 |
| transformer params | 1.00 | 1.00 | 0.8 to 1.0 |
| transformer madds | 0.4 to 0.8 | under 0.15 | under 0.05 |

In a sentence: at equal compute the vvm stores several times more pairs than any
baseline and the gap widens with the bank (N=65536 stores all 65536 pairs for about 81k
multiply-adds); at equal parameters it is level with the MLP, no better. Bank use: above
0.95 of entries with entropy above 0.9 for the 4096-entry bank at every size; for the
65536-entry bank 0.5 to 0.9 of entries at 65536 items and 0.3 to 0.6 at 16384.

**0c modular addition.** VVM (2 or 4 ticks): full training accuracy, held-out under
0.05 on every seed. With decay of the read rows: still under 0.2 (I give it 30% to reach
0.9). VVM with no bank (the tied ALU alone): 0.9 or more on at least 2 of 3 seeds (60%
sure). MLP and transformer at equal parameters: 0.95 or more. At equal compute, and the
GRU at either size: 0.9 or more on at least 2 of 3 seeds, less sure (55%).

**0c parity, 12 bits.** With weight decay 0: MLP, transformer and the vvm (whole
input, with or without the bank) at 0.45 to 0.6 held-out; the parameter-matched GRU at
0.95 or more on at least 2 of 3 seeds; the vvm fed one bit a tick at 0.95 or more on
at least one seed but not on all. With weight decay 1: not piloted; the MLP has perhaps a
40% chance of generalising, the GRU may be hurt.

**0d pointer chasing.** With the task as set (loss on the answer only): every line, at
both weight decays, under 0.05 on the held-out pairs. That is the vvm at 1, 2, 4 and 6
ticks, with the input held, with the small bank, with read-row decay, without the bank,
without the ALU, and every baseline including the GRU with extra steps. The walk does not
form on its own: the look inside the 4-tick vvm shows no diagonal; the state reads out
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

## 0d results: pointer chasing

`scripts/0d_chase.sh`, stopped after 98 of 114 runs: weight decay 1 complete (3 seeds a
line), weight decay 0 at 2 seeds a line (1 for two of the lines). 6000 steps. Train
accuracy is 1.000 for every line. The test is 240 held-out pairs, so one pair is 0.004 and
differences under about 0.02 are noise. Accuracy is the mean over seeds with the lowest
and highest seed. Bank columns: used = share of entries picked at all in a clean pass,
eff = exp(entropy of the read weight) = effective number of entries, top = the top entry's
share of the read weight.

Weight decay 1 (3 seeds):

| model | params | madds | held-out acc | used | eff | top |
|---|---|---|---|---|---|---|
| vvm T=1 | 120.8k | 40.4k | 0.007 [0.004, 0.013] | 0.628 | 398 | 0.049 |
| vvm T=2 | 120.8k | 64.2k | 0.004 [0.000, 0.008] | 0.792 | 360 | 0.028 |
| vvm T=4 (reference) | 120.8k | 111.9k | 0.008 [0.000, 0.017] | 0.985 | 650 | 0.010 |
| vvm T=6 | 120.8k | 159.5k | 0.010 [0.004, 0.021] | 0.989 | 625 | 0.013 |
| vvm T=4 hold | 120.8k | 111.9k | 0.010 [0.004, 0.017] | 0.958 | 328 | 0.090 |
| vvm T=4 N=256 | 71.2k | 107.8k | 0.011 [0.008, 0.013] | 0.943 | 26 | 0.313 |
| vvm T=4 vdecay | 120.8k | 111.9k | 0.006 [0.004, 0.008] | 0.979 | 689 | 0.006 |
| vvm T=4 no alu | 104.1k | 46.1k | 0.008 [0.004, 0.013] | 0.985 | 660 | 0.012 |
| vvm T=4 no bank | 50.1k | 82.4k | 0.125 [0.098, 0.157] | - | - | - |
| vvm T=4 tick loss | 120.8k | 111.9k | 0.365 [0.317, 0.428] | 0.947 | 170 | 0.226 |
| vvm T=6 tick loss | 120.8k | 159.5k | 0.260 [0.104, 0.406] | 0.971 | 68 | 0.351 |
| vvm T=4 hold tick loss | 120.8k | 111.9k | 0.246 [0.104, 0.343] | 0.966 | 197 | 0.203 |
| mlp params | 120.7k | 75.2k | 0.017 [0.008, 0.034] | - | - | - |
| mlp madds | 172.3k | 112.2k | 0.013 [0.008, 0.017] | - | - | - |
| gru params | 119.4k | 156.6k | 0.022 [0.017, 0.029] | - | - | - |
| gru madds | 89.5k | 110.9k | 0.013 [0.008, 0.017] | - | - | - |
| gru+4 params | 119.4k | 416.2k | 0.051 [0.038, 0.062] | - | - | - |
| gru+4 madds | 43.6k | 110.8k | 0.038 [0.021, 0.056] | - | - | - |
| transformer params | 122.2k | 186.3k | 0.033 [0.017, 0.043] | - | - | - |
| transformer madds | 80.8k | 116.2k | 0.048 [0.030, 0.058] | - | - | - |

"params" and "madds" after a baseline's name say which of the reference vvm's two
counts it was sized to. The MLP's first layer is read as embedding rows, so its compute is
below its parameter count.

Weight decay 0 (2 seeds; the full table is printed by the script line below): every line
is at 0.002 to 0.017 held-out except the tick-loss lines: T=4 0.121 [0.079, 0.162], T=4
hold 0.086, T=6 0.036. The vvm without the bank is at 0.008 there, so its 0.125 above
needs the weight decay.

Held-out accuracy by the count h, weight decay 1:

| model | h = 2 | h = 3 | h = 4 |
|---|---|---|---|
| vvm T=4 | 0.013 | 0.005 | 0.009 |
| vvm T=4 no bank | 0.122 | 0.149 | 0.104 |
| vvm T=4 tick loss | 0.607 | 0.307 | 0.180 |
| vvm T=6 tick loss | 0.455 | 0.258 | 0.057 |
| vvm T=4 hold tick loss | 0.429 | 0.217 | 0.084 |

The look inside (weight decay 1, 3 seeds, inputs with h = 4, trained and held-out pairs
together): the share of inputs whose state after the tick reads out, through the vvm's
own readout, as f applied i times to x.

VVM T=4, loss on the answer only. No walk: the answer is there after the first tick.

| tick | i = 0 | i = 1 | i = 2 | i = 3 | i = 4 |
|---|---|---|---|---|---|
| 1 | 0.014 | 0.064 | 0.046 | 0.043 | 0.647 |
| 2 | 0.013 | 0.046 | 0.040 | 0.043 | 0.691 |
| 3 | 0.012 | 0.043 | 0.035 | 0.043 | 0.691 |
| 4 | 0.012 | 0.046 | 0.030 | 0.038 | 0.691 |

VVM T=4, readout supervised after every tick. A walk, one hop a tick:

| tick | i = 0 | i = 1 | i = 2 | i = 3 | i = 4 |
|---|---|---|---|---|---|
| 1 | 0.004 | 0.997 | 0.004 | 0.014 | 0.012 |
| 2 | 0.010 | 0.023 | 0.974 | 0.005 | 0.010 |
| 3 | 0.010 | 0.022 | 0.049 | 0.931 | 0.004 |
| 4 | 0.012 | 0.017 | 0.042 | 0.194 | 0.743 |

Fitted decoder scored on held-out start items, per tick: 0.07, 0.21, 0.37, 0.52 without
the tick loss; 0.74, 0.75, 0.76, 0.71 with it.

Against the predictions: "every line under 0.05 with the task as set" held for the vvm
lines with a bank and for the MLP, the GRU and the transformer; it was wrong for the
vvm without its bank (0.125) and marginal for the GRU with extra steps (0.051). "No
diagonal" held. "Tick loss reaches 0.8 or more" was wrong (0.365), and so was "if not,
under 0.2". "Holding the input costs 0.2 or more under the tick loss" was wrong: it costs
0.12, inside the seed range.

What is suspect:

- The look-inside tables mix trained and held-out pairs (about 70 to 30). The diagonal
  under the tick loss is therefore mostly trained pairs; a table for the held-out pairs
  alone was not built. From the shares, the held-out pairs seem to follow the walk for the
  first ticks and fail on the last hop, but that is arithmetic on a mixed table, not a
  measurement.
- The tick loss is not the task as set. It shows the vvm can hold a walk when asked
  for one at every tick; it does not show the walk forming.
- Under the tick loss one entry takes 0.2 to 0.35 of all read weight. I did not look at
  what that entry is (a do-nothing entry for the ticks after the count is a guess).
- One bank size, one k, one learning rate, 6000 steps. The dense tied block alone beating
  the full vvm (0.125 against 0.008) is 3 seeds at one setting: a hint.
- Bank use here is not a failure sign by itself: every line fits its training pairs. The
  one line that may be locked in is N=256 at weight decay 1 (26 effective entries of 256,
  the top entry at 0.31), and it still fits the training pairs.

## What was not done

- **0b, 0c, 0e scoring runs.** `scripts/0b_assoc.sh`, `scripts/0c_compute.sh` and
  `scripts/0e_collapse.sh` are written and were never run. Their smoke-sized versions run.
- **0d**: the third seed at weight decay 0; the look inside for held-out pairs alone.
- **The two bank measurements asked for during the work**: entries used against task size
  (the 0b script varies the number of items at a fixed bank and the table prints used and
  eff, but it was not run, and it has no sizes below the bank's size); how much the picked
  entries change with the input (mean overlap of the top-k sets of two random inputs): not
  built.
- **Noise as asked for during the work**: there is no noise on the value as read, and the
  state noise is scaled by the state's rms instead of being a plain strength. The query
  noise, the Gumbel noise and pick_from are plain numbers already.
- **Data memory**: version 0 is built (`vvm2/memory.py`) and runs in the smoke test on a
  small key-value recall task; nothing was measured with it.
- **More than one register**: runs in the checks and the smoke test only.

## Where this differs from docs/design.md, or where the design did not say

1. **Compute grows with the bank.** Scoring the sub-keys costs heads x sqrt(N) x key width
   per tick. It is small here (4096 of about 26k multiply-adds a tick at N = 4096) but it
   is not constant: at N = 2^30 with 2 heads and key width 32 it is about 2M a tick, far
   more than the ALU. The README's "the cost of a step does not grow with the size of the
   bank" is wrong for product keys as designed. `vvm2/check.py` pins the exact growth.
2. **Input write.** "Written on every tick it is present" does not say when one fixed
   input is present. If it is re-written on every tick, a query that is a fixed linear map
   of the state cannot follow a pointer, because the start item keeps being added back.
   I chose: the input is added to register 0 on the first tick of that input only
   (`--hold 1` gives every tick). In 0d the two were the same on the task as set and
   holding was a little worse under the tick loss.
3. **Write means add.** A set would erase a one-register vvm each tick.
4. **Registers.** The design does not say which register a value edits. I chose: the
   query and the ALU see all registers; head h adds its value (width d) to register
   h mod R; the input goes to register 0, a data-memory read to the last register.
5. **Heads.** One bank: heads are separate queries into the same sub-keys and values.
6. **Query norm.** Layer norm without gain over each head's query (`--qnorm` has l2,
   batch and none).
7. **Readout.** A layer norm before the linear readout.
8. **Sparse training.** Values are an `nn.Embedding(sparse=True)`; only the rows read get
   a gradient (checked); `torch.optim.SparseAdam` updates them. Two costs: SparseAdam
   keeps two moment tables the size of the bank, which will not do for a bank on disk; and
   the optimiser's weight decay cannot reach the bank. `--vdecay` shrinks the rows read on
   a step (not in the design). The sub-key tables are small and trained by AdamW.
9. **No way to stop.** Chasing with a count needs the vvm to do nothing after h
   ticks. Nothing in the design counts ticks or halts; the state has to carry a counter.
10. **Data memory.** Version 0 scans every slot on a read and appends one slot a tick;
    the design's "same top-k lookup" is honoured only in the top-k, not in the addressing.
11. **Accounting.** One multiply-add per weight use in a matrix product, the row width per
    embedding or value read, the vector width per norm; biases and activations free. For
    one-shot tasks "per step" is per example, all ticks.
12. **Added, not in the design**: the tick loss (`--tick_loss`), the GRU with extra steps
    on a zero input, read-row decay.

## Proposed edits (not made)

`docs/findings.md`, all at grade **hint**:

- Pointer chasing, 256 items, counts 1 to 4, 30% of pairs held out: with the loss on the
  answer only, the vvm at 1 to 6 ticks and all three baselines fit the training pairs
  and stay at chance on held-out pairs; the answer is in the state after the first tick.
  (docs/results-2026-11-26-foundation.md)
- With the readout supervised after every tick the vvm walks f one hop a tick on
  trained pairs and gets 0.37 of held-out pairs (0.61 at 2 hops, 0.18 at 4), with weight
  decay 1; 0.12 without weight decay.
- The dense tied block alone, with weight decay, gets 0.125 of held-out pairs where the
  vvm with its bank gets 0.008: the bank may be what lets it store pairs instead of
  walking.

`docs/plan.md`:

- 0a: mark landed. 0b, 0c, 0e: scripts and predictions exist, runs queued. 0d: first
  numbers in; open question is why the held-out pairs fall off the supervised walk.
- 0e: reword after the owner's note: low use is not the fault; being locked in is. Add
  the two measurements (entries used against task size, overlap of the picked sets of two
  inputs) and noise on the value as read.
- Phase 0 gate or design: note that product-key scoring costs sqrt(N) a lookup, and
  decide what addresses a bank of 10^9 entries.

## Script lines

```
scripts/setup.sh
scripts/smoke.sh
scripts/pilot.sh store
scripts/pilot.sh modadd
scripts/0d_chase.sh                    # ran, stopped at 98 of 114; a rerun fills the gap
.venv/bin/python -m vvm2.table out/0d_chase.jsonl --split wd
.venv/bin/python -m vvm2.table out/0d_chase.jsonl --split wd --groups
.venv/bin/python -m vvm2.table out/0d_chase.jsonl --split wd --lens "vvm T=4 tick loss"
scripts/0b_assoc.sh                    # not run
scripts/0c_compute.sh                  # not run
scripts/0e_collapse.sh                 # not run
```
