# Ordinary curves, then control

2026-11-27. The machine is `vvm2/core.py` as frozen (exact pointers, steps and seek). The
tasks are `vvm2/curves.py` and `vvm2/control.py` (worlds in `vvm2/worlds.py`); the scripts
are `scripts/curves.sh` and `scripts/control.sh`.

## Predictions (written and committed before any scoring run; not edited after)

Nothing had been run when these were written: no pilot, no smoke run.

**Curves.** Machine: bank of 16, 4 fetched, 4 ticks, 2 read pointers (about 115k weights),
3,000 steps, beside an MLP of the same weights.

| task | MLP | machine, layout 'one' | machine, layout 'each' | run with whole steps |
|---|---|---|---|---|
| xor, noisy test points | 0.99 or more | 0.99 or more | 0.99 or more | the same as trained |
| sine, mean squared error | 0.0005 or less | 0.001 to 0.01, worst at the two peaks | the same layout (one number) | the same as trained |
| two spirals, test | 0.99 or more | 0.85 to 0.97 | 0.80 to 0.97 | within 0.02 of trained |

- The pointers will have nothing to do: the read pointers sit on the input and the write
  pointer on the output from the first tick, and the run is a small recurrent net through
  the pc slot. So whole steps should cost nothing.
- With one read pointer and layout 'each' (it must step to see the second number) xor is
  still learned; spirals lose some.
- The machine will be behind the MLP on the sine and on spirals at the same weights: a tick
  gives it 4 blended clipped-linear maps, and its first tick fetches the same vops whatever
  the input.

**Balancing a pole, evolution strategy, whole steps.** Machine: bank of 4, 2 fetched, 1
tick a world step (about 29k weights).

- The machine balances: mean of 100 test episodes above 450 of 500, on at least 2 of 3
  seeds, inside 10,000 episodes of experience.
- The MLP and the GRU do the same. The MLP gets there in the fewest episodes; the machine
  takes the most (it first has to find a write pointer that is on the output slot).
- The trace is a reflex: read pointers on the input, write pointer parked on the output,
  scratch never written, one or two vops. A machine started fresh on the present state
  pushes the same way on 0.95 or more of steps.

**Policy gradient through the soft machine.** Learns something but worse than the evolution
strategy for the same episodes: 100 to 400 steps as trained, and lower run with whole
steps. The MLP by the same policy gradient does better than the machine.

**Velocities hidden.** The MLP cannot (under 100 steps). The GRU can (over 400). The
machine lands between, 100 to 500, and when it does well it keeps the last position in the
pc slot, not in scratch.

**Swing-up.** With this evolution strategy and budget none of the three gets a clean
swing-up on every start; the machine is no better than the GRU.

**What the machine brings to control that an MLP does not:** on a fully seen pole, nothing.

## The answers, a sentence each

1. **Curves.** As trained the machine fits all three like an MLP of the same weights
   (spirals 1.000, sine error 0.00002, xor 1.000 on the corners and 0.97 on noisy points);
   run with whole steps the same weights fail (xor 0.79, sine error 0.45, spirals 0.56),
   because training leaves the pointers half on one slot and half on another and the fit
   leans on that blur. Adding the whole-step run of each batch to the training loss fixes
   it (sine 0.00002, spirals 0.985 to 0.988, xor 0.94 with whole steps). Both layouts
   behave alike.
2. **Balancing a pole.** The machine, trained by an evolution strategy with whole steps and
   no labels, balances all 500 steps on 100 of 100 test episodes on 6 of 6 runs; it takes
   about 3 times the episodes an MLP of the same weights takes (9,720 against 3,312; 2.2 to
   4.6 times by seed) and about twice a GRU's. It
   never uses scratch; it is not a pure reflex either, because the pc slot carries the last
   few world steps into each push. A policy gradient through the soft
   machine (one seed) also balances, 500 of 500 as trained and with whole steps, in about a
   sixth of the experience (1,600 episodes; the MLP by the same gradient 320).
3. **Hidden velocities, swing-up:** not run (stopped at the coordinator's word). The worlds
   and script lines exist and start without error; no scoring run was made or read.
4. **What the machine brings to control that an MLP does not, on this evidence:** nothing
   for a pole. It balances, at half the multiply-adds a step of the same-weights MLP, and
   costs about 3 times the experience under evolution. It has memory the MLP lacks and a pole seen whole
   does not need it.

## Curves

How the numbers get in and out is in the header of `vvm2/curves.py`: raw numbers in the
first places of one input slot (layout 'one') or one slot a number (layout 'each'), the
rest 0, no learned encoder; the answer is the first two numbers of the one output slot
(classes: the larger wins) or its first number (sine). Machine: bank of 16, 4 fetched, 4
ticks, 2 read pointers, 4 scratch slots. 3,000 steps, batch 64, AdamW at 0.002, one-cycle,
the loop in `vvm2/common.py`, the same for the MLPs. 3 seeds; mean (each seed). Weights:
machine 114,692; MLP of the same weights 114,914; MLP of the same multiply-adds 113,562.
Multiply-adds a run: 113,664; 114,914; 113,562. (At this bank size the two matches come to
the same MLP within 1%, because the machine fetches a quarter of its bank on each of 4
ticks.)

Test scores (xor: 1,000 noisy copies of the corners; sine: mean squared error on a grid;
spirals: 1,000 fresh points). Train scores are 1.000 for every class row except where said.
"Both" is the machine trained on whole steps too.

| task | layout | as trained | the same weights, whole steps | both: as trained | both: whole steps | MLP, same weights | MLP, same multiply-adds |
|---|---|---|---|---|---|---|---|
| xor | one | 0.972 (0.993, 0.943, 0.979) | 0.791 (0.810, 0.838, 0.725); train 0.83 | 0.961 (0.983, 0.908, 0.992) | 0.942 (0.960, 0.899, 0.966) | 0.998 | 0.997 |
| xor | each | 0.965 (0.953, 0.963, 0.980) | 0.475 (0.521, 0.499, 0.406); train 0.42 | 0.953 (0.972, 0.963, 0.923) | 0.945 (0.870, 0.998, 0.967) | 0.998 | 0.997 |
| sine | one | 0.00002 (all three) | 0.452 (0.302, 0.490, 0.563) | 0.00003 (all three) | 0.00002 (0.00001, 0.00002, 0.00001) | under 0.000005 | under 0.000005 |
| spirals | one | 1.000 (0.999, 1.000, 1.000) | 0.555 (0.402, 0.455, 0.808) | 1.000 | 0.985 (1.000, 0.955, 1.000); train 0.987 | 1.000 | 1.000 |
| spirals | each | 1.000 (0.999, 1.000, 1.000) | 0.504 (0.450, 0.512, 0.550); train 0.516 | 0.999 (0.999, 0.999, 1.000) | 0.988 (0.965, 0.999, 1.000); train 0.989 | 1.000 | 1.000 |

"Trained on whole steps too" (`mix=1`): every training step adds the loss of the same
batch run with whole steps. That run is not differentiable in the pointer moves, so its
gradient reaches the stored values and the pc (and so the fetch) only; the moves still
learn from the soft run.

**Why whole steps break a plainly trained machine.** The pointers are exact to 0.47 to 0.65
on these tasks (on copy and reverse they were about 0.9). Nothing in a curve needs a
pointer to be anywhere in particular, so a pointer spread over the input slot and the
output slot is just one more blend the gradient can use, and the fit ends up depending on
the shares. Whole steps put each pointer on its likeliest slot and the function changes.
On the list tasks the task itself pressed the pointers to be exact; here nothing does.

**Layouts.** 'one' and 'each' score the same as trained. Neither works with whole steps
unless trained for it, and both work when trained for it. No layout failed on its own
account. (With 2 read pointers 'each' needs no stepping: the pointers can start on the two
slots.)

**The small gap on xor's noisy points** (0.94 to 0.97 against 0.998) is on the test only:
training has just the 4 corners, and between them the machine's answer is set by which
vops the pc fetches, which changes in jumps, where the MLP's boundary is smooth. That is a
reading, not a measurement. On spirals, with 1,000 training points, there is no gap.

**Bank use** (run with whole steps, on the test set; layout 'one', plain training): xor
0.25 to 0.56 of 16 vops used, entropy 0.32 to 0.63, busiest 0.32 to 0.68, 8 to 15
different paths; sine 0.31 to 0.38, entropy 0.42 to 0.54, 8 to 10 paths; spirals 0.62 to
0.75, entropy 0.60 to 0.75, 34 to 49 paths, the commonest taken by 0.09 to 0.16 of points.
So the harder curve draws on more of the bank and the path depends on the input. The first
tick always fetches the same vops (the pc starts the same for every input).

**Trace** (xor, seed 0, corner 0,1, whole steps, trained on whole steps too, layout
'each'): after tick 1 (vop 14) one read pointer has sought to the second input slot and
the other read pointer and the write pointer are on the output slot; after ticks 2 to 4
(vops 3, 12, 13) all three pointers are on the output slot, rewriting it. Other corners take other vops after the first tick. It is a small recurrent
net steered by the pc: read the input once, then work on the output slot in place.

## Balancing a pole

**The world** (`vvm2/worlds.py`, written here, batched, torch only): the standard cart and
pole. Gravity 9.8, cart mass 1.0, pole mass 0.1, pole half-length 0.5, push 10 N left or
right, world step 0.02 s by Euler's rule, start with all four state numbers uniform in
+-0.05, over when the cart is 2.4 from the middle or the pole 12 degrees from upright,
capped at 500 steps. Steps balanced counts the step that drops it.

**The loop** (`vvm2/control.py`): the driver writes cart position, cart velocity, pole
angle and pole angular velocity raw into the first four places of one input slot before
every tick; the cart is pushed right when the first number of the output slot is larger
than the second. One tick a world step. The machine is started once an episode and nothing
is wiped until the episode ends. Machine: bank of 4, 2 fetched, 2 read pointers, 4 scratch
slots: 28,699 weights, 14,144 multiply-adds a world step.

**What the core lacks for this.** `Core.forward` calls `world(t)` for the input and does not
give it the output slots, and a control world needs the output to make the next state; nor
can the world end a run. The core was not changed. `control.py` reads the core's memory out
of the running `forward` through its Python frame (`machine_memory`) and raises an
exception to stop when every episode is over. Proposed for the core when it is next open:
call `world(t, out)` with the output slots, and stop when it returns None.

**Training.** Evolution strategy, forward runs only, the machine with whole steps
throughout: each round 32 noisy copies of the weights (16 mirrored pairs), each scored on
the same 4 fresh episodes; the weights move by `noise size` times the rank-weighted mean of
the noise; the unperturbed weights are then checked on 16 fresh episodes; stop when every
check episode has run to 500 three rounds running, or at 200 rounds. No labels and no
hand-written controller anywhere. The MLP (two hidden layers) and the GRU (one GRU step a
world step) get the same four raw numbers and the same code.

Test: 100 fresh episodes. Every run below scored 500.0 with 100 of 100 episodes reaching
500, except one (the MLP of the same multiply-adds, noise 0.1, seed 2: 498.9, 97 of 100).
Episodes of experience until the unperturbed weights first averaged 475 on the 16 check
episodes, seeds 0, 1, 2:

| model | weights | multiply-adds a world step | noise 0.1 | noise 0.05 | mean of the six |
|---|---|---|---|---|---|
| vvm, whole steps | 28,699 | 14,144 | 9,936; 9,072; 11,952 | 12,816; 6,768; 7,776 | 9,720 |
| MLP, same weights | 28,547 | 28,547 | 2,160; 3,456; 4,608 | 4,320; 3,024; 2,304 | 3,312 |
| GRU, same weights | 28,977 | 28,977 | 6,624; 9,072; 2,736 | 6,048; 5,472; 1,728 | 5,280 |
| MLP, same multiply-adds | 14,147 | 14,147 | 4,896; 3,456; 5,760 | 4,032; 3,600; 4,608 | 4,392 |

Screens, one seed (0), the machine alone, noise 0.05: noise 0.02 learns too slowly (43.5
steps after 28,800 episodes); 2 ticks a world step balances (500.0, first at 11,232
episodes, twice the multiply-adds); layout 'each' (4 input slots, 2 read pointers)
balances (499.9, 99 of 100, first at 14,544), with one read pointer staying on the pole's
angular velocity and the other seeking back and forth among the other slots.

**Policy gradient through the soft machine** (one seed; 16 episodes a round, Adam at 0.003,
the push sampled from the two output numbers, reward-to-go less a per-step average, the
gradient going back through the whole episode):

| model | weights | steps balanced, 100 test episodes | share reaching 500 | episodes to first reach 475 (checked every 160) | episodes used |
|---|---|---|---|---|---|
| vvm, as trained (soft) | 28,699 | 500.0 | 1.00 | 1,600 | 3,200 |
| vvm, the same weights with whole steps | 28,699 | 500.0 | 1.00 | | |
| MLP, same weights | 28,547 | 500.0 | 1.00 | 320 | 3,200 |

The machine's learning was not steady: 500 on the check at 1,600 and 1,920 episodes, 220
at 2,240, 500 again from 2,560 to the end. Whole steps cost nothing here because the
pointers end up parked (read on input, write on output, one vop, scratch never written;
fresh-machine agreement 0.68 at 1 state, 0.99 at 32), the same shape the evolution
strategy found.

**What the trained machine does, tick by tick** (noise 0.1, seed 0; the first ticks of an
episode it balanced for 500 steps; `i0` the input slot, `o0` the output slot):

```
vop:        0     0     0     0     0     0     0     0     0     0     0     0  ...
read 1:    i0    i0    i0    i0    i0    i0    i0    i0    i0    i0    i0    i0  ...
  did:    inN   in0   in0   in0   in0   in0   in0   in0   in0   in0   in0   in0
read 2:    i0    i0    i0    i0    i0    i0    i0    i0    i0    i0    i0    i0  ...
  did:    inN   inN   in0   inN   in0   inN   in0   inN   in0   inN   in0   inN
write:     o0    o0    o0    o0    o0    o0    o0    o0    o0    o0    o0    o0  ...
  did:   outN  outN  out0  out0  out0  out0  out0  out0  out0  out0  out0  out0
push:       L     R     L     R     L     R     L     R     L     R     L     R  ...
```

Over 16 episodes of 500 steps, the six evolution-strategy machines:

| noise, seed | vops used of 4 (busiest share) | read pointers on | write pointer on | largest number ever in scratch | pc slot moves a world step | fresh machine, 1 state | 2 | 4 | 8 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1, 0 | 1 (1.00) | input, input | output | 0 | 0.20 | 0.904 | 0.937 | 0.992 | 0.998 | 0.999 |
| 0.1, 1 | 1 (1.00) | input, output | output | 0 | 0.26 | 0.729 | 0.886 | 0.902 | 0.943 | 0.996 |
| 0.1, 2 | 1 (1.00) | input, input | output | 0 | 0.30 | 0.849 | 0.760 | 0.772 | 0.952 | 0.998 |
| 0.05, 0 | 4 (0.98) | input, input | output | 0.65 (first ticks only) | 0.30 | 0.777 | 0.787 | 0.770 | 0.875 | 0.974 |
| 0.05, 1 | 1 (1.00) | input; scratch on 0.78 of ticks (it is empty) | output | 0 | 0.24 | 0.865 | 0.807 | 0.919 | 0.960 | 0.988 |
| 0.05, 2 | 3 (0.68) | input, input | output | 0 | 0.45 | 0.755 | 0.856 | 0.836 | 0.962 | 0.996 |

The last five columns are the reflex test: a machine started fresh is shown only the last
1, 2, 4, 8 or 32 states of the real episode (the oldest held for 8 ticks first), and the
cell is the share of about 8,000 steps on which it pushes the same way the running machine
did. A pure reflex would score 1.0 in the first of them.

So: the read pointers park on the input, the write pointer parks on the output, one vop
runs on nearly every tick, and scratch is never used (one machine wrote to it in its first
ticks and never read it back; one keeps a read pointer on empty scratch). It is not a pure
reflex: the pc slot is memory, it moves 0.2 to 0.45 of its length every world step, and the
push depends on it. Shown only the present state a fresh machine agrees on 0.73 to 0.90 of
steps; shown the last 32 states, on 0.97 to 0.999. So each push is a function of the last
several states, held in the pc slot (and for one machine also in the output slot, which a
read pointer reads back). Whether it needs that memory to balance cannot be tested without
wiping the pc slot mid-run, which the frozen core gives no way to do; the MLP shows the
task does not need it.

Bank use on the pole: 1 of 4 vops on 4 of 6 machines. With one vop fetched as the best
match every tick, the pc decides nothing and the machine is one matrix applied again and
again to (input slot, output slot, pc slot): a small recurrent net. That is fine here (the
score is perfect) and it is not evidence about the bank. (The fetch still blends 2 vops by
nearness; only the best match is counted.)

## Predictions against results

- xor 0.99 or more for the machine: **wrong**, 0.97 as trained (the corners are all right;
  the noisy points are not).
- sine 0.001 to 0.01 for the machine: **wrong, it did better**, 0.00002. Still behind the
  MLP (under 0.000005).
- spirals 0.85 to 0.97: **wrong, it did better**, 1.000, level with the MLP.
- Whole steps the same as trained, the pointers having nothing to do: **wrong**, and this
  is the main thing learned about curves. Having nothing to do, the pointers stay blurred,
  and the blur is used.
- One read pointer with layout 'each': not run.
- The machine balances on at least 2 of 3 seeds inside 10,000 episodes: **right, barely**
  (2 of 3 at each noise size; all 6 balance by 12,816).
- The MLP is quickest and the machine slowest: **right** on the means; the GRU beat the MLP
  on one seed of three at each noise size.
- The trace is a reflex, scratch never written, a fresh machine agreeing on 0.95 or more:
  **half right**. Scratch is unused and the pointers park, but a fresh machine agrees on
  only 0.73 to 0.90: the pc slot carries the recent past.
- Policy gradient worse than the evolution strategy for the same episodes: **wrong**. On the one seed it
  balanced in 1,600 episodes against 6,800 to 12,800 by evolution, and whole steps did not
  hurt. The MLP did do better by the same gradient (320 episodes).
- Hidden velocities, swing-up: not run.

## What is suspect

- The evolution strategy's settings (32 copies, 4 episodes each, the two noise sizes) were
  not tuned for any model; the order MLP, GRU, machine could move with other settings. The
  spread across seeds is as large as the gap between GRU and MLP.
- "Episodes to first reach 475" is read off 16 check episodes a round, so it is coarse
  (steps of 144 episodes) and a lucky round can set it.
- The same-multiply-adds comparison counts only fetch and execute for the machine, as
  ticks x (bank x 16 + k x matrix). Pointer bookkeeping and the seek are not counted.
- The reflex test starts a fresh machine and holds the oldest state for 8 ticks; the
  running machine got to its state another way. That agreement reaches 0.97 to 0.999 at 32
  states says the test is sound, but the low numbers at 1 state could partly be the fresh
  machine's start-up and not history.
- No transformer is in any table; AGENTS.md asks for one beside every claim. An MLP and a
  GRU were what the task named.
- "Trained on whole steps too" was tried once, at one weighting (the two losses added).
- The policy gradient is one seed.
- The closed loop works by reading the core's memory from a Python frame. It reads only,
  but it depends on the name `mem` inside `Core.forward`.

## What was not done

- Item 3, hidden velocities and the swing-up: skipped at the coordinator's word. The code
  for both is written (`scripts/control.sh blind 150`, `scripts/control.sh swing 150`).
  Each was started once for 2 tiny rounds to see that it does not crash; the scores were
  not read and mean nothing at that size.
- More ticks, bigger or smaller banks, one read pointer, `gain=2` on the sine: options
  exist, none was run.
- A press toward exact pointers as a training signal (the core's own measure, `sharp()`,
  is detached, so it cannot be used as a loss without opening the core).
- Wiping the pc slot in the middle of an episode, to see whether the pole machine needs
  its memory.
- The machine with a bigger bank on the pole, or several behaviours in one bank, which is
  where a bank might matter for game characters.

## The exact script lines

```
scripts/setup.sh
scripts/curves.sh all 3000 seeds=3                        # layout 'one', the main table
scripts/curves.sh all 3000 seeds=3 mix=1                  # trained on whole steps too
scripts/curves.sh xor 3000 seeds=3 layout=each
scripts/curves.sh spirals 3000 seeds=3 layout=each
scripts/curves.sh xor 3000 seeds=3 layout=each mix=1
scripts/curves.sh spirals 3000 seeds=3 layout=each mix=1
scripts/control.sh pole 200 seed=0 sigma=0.1 who=vvm,mlp,gru,mlp2     # and seed=1, seed=2
scripts/control.sh pole 200 seed=0 sigma=0.05 who=vvm,mlp,gru,mlp2    # and seed=1, seed=2
scripts/control.sh pole 200 seed=0 who=vvm sigma=0.02                 # screens
scripts/control.sh pole 200 seed=0 who=vvm ticks=2
scripts/control.sh pole 200 seed=0 who=vvm layout=each
scripts/control.sh pole 200 train=pg seed=0 who=vvm,mlp
scripts/curves.sh --quick
scripts/control.sh --quick
```

The first two curves lines and every evolution run on the pole but noise 0.05 seed 2 were
made with 4 threads, the rest with 1; the code now sets 1.

## Proposed edits (not made)

**docs/findings.md**, new entries:

38. **The ordinary curves are learned on the exact-pointer machine, level with an MLP, but
    not with whole steps unless it is trained for them.** [likely: 3 seeds,
    `scripts/curves.sh all 3000 seeds=3`, `... mix=1`] Numbers laid raw into one input
    slot, the answer read from one output slot, 4 ticks, bank of 16, 115k weights. As
    trained: spirals 1.000 (MLP 1.000), sine error 0.00002 (MLP under 0.000005), xor 1.000
    on the corners and 0.97 on noisy copies (MLP 0.998). The same weights with whole
    steps: xor 0.79, sine 0.45, spirals 0.56. The pointers are only half exact (0.47 to
    0.65): a curve gives them nothing to do, so they stay blurred and the fit uses the
    blur. Adding the whole-step run of every batch to the loss gives sine 0.00002, spirals
    0.985, xor 0.94 with whole steps. One slot for all the numbers and one slot a number
    score alike.
39. **The machine balances a pole, trained by an evolution strategy with whole steps and no
    labels.** [solid: 3 seeds at each of two noise sizes, `scripts/control.sh pole 200
    seed=N sigma=0.1 who=vvm,mlp,gru,mlp2`] 500 of 500 steps on 100 of 100 test episodes
    on all six runs, 29k weights, bank of 4, one tick a world step. Episodes of experience
    to get there: machine 9,720, GRU 5,280, MLP 3,312 (means of six; the MLP with the
    machine's multiply-adds 4,392). It brings nothing to this task that an MLP does not.
40. **The pole machine parks its pointers and never uses scratch; its memory is the pc
    slot.** [likely: six machines] Read pointers on the input, write pointer on the
    output, one vop on 4 of 6 machines. A machine started fresh on the present state
    pushes the same way on 0.73 to 0.90 of steps, and on 0.97 to 0.999 when shown the last
    32 states: the push depends on the recent past through the pc slot.
41. **The core cannot close a loop with a world on its own.** `world(t)` is not handed the
    output slots and cannot stop the run. Worked round from outside for control
    (`vvm2/control.py`, `machine_memory`); the core should call `world(t, out)`.
42. **A policy gradient through the soft machine balances the pole in a sixth of the
    episodes evolution takes, and the result runs with whole steps unchanged.** [hint: one
    seed, `scripts/control.sh pole 200 train=pg seed=0 who=vvm,mlp`] 1,600 episodes
    against 9,720; an MLP by the same gradient, 320. Learning dipped once and recovered.

**docs/plan.md**, the battery line "Ordinary curves and control": *(done: curves and the
pole, findings 38 to 42; open: hidden velocities, swing-up)*.

**docs/ideas.md**, new entries:

63. **Train on the whole-step run as well** (from finding 38): add the loss of the same
    batch run with whole steps. Where a task gives the pointers nothing to do, the soft
    machine fits with blurred pointers and breaks when run exactly. Worth trying on the
    lights game and lookup, which also lean on blur (findings 34, 36).
64. **A press toward exact pointers**: `sharp()` as a loss term. Needs the core to keep the
    undetached number.
65. **`world(t, out)`, and a world that can stop the run**: the closed-loop form of the
    world function, for control, games and anything that answers back.
66. **Evolution on the whole-step machine as the ordinary way to train characters**
    (owner's interest): it works on the pole with forward runs only, at about 3 times an
    MLP's experience. The test that matters next is one an MLP cannot do: velocities
    hidden, so the last position must be kept (pc slot or scratch), then a task that needs
    a count or a place kept for many steps, where scratch and pointers should earn their
    keep.
67. **Many behaviours, one bank**: a pole needs one vop. Whether a bank helps control shows
    only when one machine must hold several behaviours and the pc picks between them.
