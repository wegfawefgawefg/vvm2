# Small algorithms and one conditional (2026-11-27)

Code: `vvm2/algos.py` (the ladder and the conditional). Script: `scripts/algos.sh`. The core
is not changed by any of this (the faster tick was taken from main part way through; same
numbers). The lights game (part C) was moved to another session before any run of it was
made here; the options added to `vvm2/tape.py` for it are listed at the end.

Every run: trained on lengths 2 to 8, 3,000 steps, a bank of 128 vops, 4 fetched a tick,
tested at lengths 8, 16, 32 and 64 on 300 fresh inputs each. A cell is share of items right
/ share of whole answers right. "Whole steps" is the trained machine run with
`core.hard = True`. The machine has 917,284 weights and does about 35,000 multiply-adds a
tick; "GRU, same weights" has 910,088 weights (599,136 multiply-adds a step), "GRU, same
arithmetic" has about 55,000 weights (34,656 multiply-adds a step).

## A. The ladder

One seed (seed 0) unless a row says otherwise: these are screens, not results.

| # | task | model | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 1 | shift | vvm, as trained | 1.00 / 1.00 | 0.88 / 0.52 | 0.42 / 0.00 | 0.21 / 0.00 |
| | | vvm, whole steps | 0.49 / 0.23 | 0.31 / 0.11 | 0.26 / 0.08 | 0.28 / 0.10 |
| | | GRU, same weights | 1.00 / 1.00 | 0.41 / 0.00 | 0.25 / 0.00 | 0.17 / 0.00 |
| | | GRU, same arithmetic | 0.95 / 0.67 | 0.33 / 0.00 | 0.22 / 0.00 | 0.16 / 0.00 |
| 2 | double | vvm, as trained | 1.00 / 1.00 | 1.00 / 1.00 | 0.94 / 0.02 | 0.65 / 0.00 |
| | | vvm, whole steps | 1.00 / 0.98 | 1.00 / 0.98 | 1.00 / 0.92 | 1.00 / 0.93 |
| | | GRU, same weights | 1.00 / 0.97 | 0.46 / 0.00 | 0.30 / 0.00 | 0.21 / 0.00 |
| | | GRU, same arithmetic | 0.89 / 0.28 | 0.41 / 0.00 | 0.25 / 0.00 | 0.18 / 0.00 |
| 3 | keep | vvm, as trained | 0.93 / 0.75 | 0.54 / 0.01 | 0.37 / 0.00 | 0.30 / 0.00 |
| | | vvm, whole steps | 0.44 / 0.09 | 0.32 / 0.00 | 0.31 / 0.00 | 0.27 / 0.00 |
| | | GRU, same weights | 1.00 / 0.99 | 0.80 / 0.37 | 0.49 / 0.00 | 0.37 / 0.00 |
| | | GRU, same arithmetic | 1.00 / 0.99 | 0.77 / 0.32 | 0.48 / 0.00 | 0.36 / 0.00 |
| 4 | count | vvm, as trained | 1.00 / 0.96 | 0.93 / 0.20 | 0.91 / 0.04 | 0.91 / 0.01 |
| | | vvm, whole steps | 0.62 / 0.02 | 0.56 / 0.00 | 0.53 / 0.00 | 0.52 / 0.00 |
| | | GRU, same weights | 1.00 / 1.00 | 0.91 / 0.23 | 0.65 / 0.00 | 0.54 / 0.00 |
| | | GRU, same arithmetic | 1.00 / 1.00 | 0.92 / 0.25 | 0.73 / 0.00 | 0.62 / 0.00 |
| 5 | twice | vvm, as trained | 0.83 / 0.01 | 0.37 / 0.00 | 0.26 / 0.00 | 0.20 / 0.00 |
| | | vvm, whole steps | 0.17 / 0.00 | 0.14 / 0.00 | 0.13 / 0.00 | 0.13 / 0.00 |
| | | GRU, same weights | 0.91 / 0.35 | 0.31 / 0.00 | 0.20 / 0.00 | 0.16 / 0.00 |
| | | GRU, same arithmetic | 0.67 / 0.01 | 0.28 / 0.00 | 0.18 / 0.00 | 0.14 / 0.00 |

For count, read the second number: a tally of about half the slots gets most slots right
whatever the count is (the first quarter are nearly always marked and the last quarter
nearly always empty).

The same machine with its moves by content held off (`seek=0`; see "What held it back"):

| # | task | model | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 1 | shift, seed 0 | vvm, as trained | 1.00 / 1.00 | 1.00 / 1.00 | 0.97 / 0.51 | 0.60 / 0.00 |
| | | vvm, whole steps | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| | shift, seed 1 | vvm, as trained | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.62 | 0.88 / 0.02 |
| | | vvm, whole steps | 0.99 / 0.97 | 0.98 / 0.92 | 0.97 / 0.78 | 0.96 / 0.76 |
| | shift, seed 2 | vvm, as trained | 1.00 / 1.00 | 1.00 / 1.00 | 0.98 / 0.65 | 0.66 / 0.00 |
| | | vvm, whole steps | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| 3 | keep | vvm, as trained | 0.84 / 0.54 | 0.55 / 0.01 | 0.40 / 0.00 | 0.33 / 0.00 |
| | | vvm, whole steps | 0.80 / 0.48 | 0.50 / 0.01 | 0.37 / 0.00 | 0.32 / 0.00 |
| 4 | count | vvm, as trained | 1.00 / 1.00 | 0.90 / 0.08 | 0.74 / 0.00 | 0.75 / 0.00 |
| | | vvm, whole steps | 0.77 / 0.21 | 0.82 / 0.10 | 0.82 / 0.05 | 0.74 / 0.00 |
| 5 | twice | vvm, as trained | 0.89 / 0.33 | 0.72 / 0.00 | 0.56 / 0.00 | 0.43 / 0.00 |
| | | vvm, whole steps | 0.78 / 0.28 | 0.75 / 0.08 | 0.71 / 0.01 | 0.67 / 0.00 |
| 6 | parity | vvm, as trained | 0.95 / 0.95 | 0.45 / 0.45 | 0.50 / 0.50 | 0.51 / 0.51 |
| | | vvm, whole steps | 0.55 / 0.55 | 0.50 / 0.50 | 0.47 / 0.47 | 0.55 / 0.55 |
| | | GRU, same weights | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| | | GRU, same arithmetic | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

Where it stops, as the machine is: after double. Shift is learned as a loop but its start is
not exact; keep, count and twice are not routines at all (three failures in a row, so the
climb stopped there; compare, adding, brackets and sort were not run). On these five tasks
no GRU carries anything past 16. On parity, one rung higher, both GRUs are exact at 64 and
the machine is at chance past 8: the one place here where the plain net carries a routine
and the machine does not.

| task | learned at trained lengths | carries, run with whole steps | carries with moves by content held off |
|---|---|---|---|
| shift | yes | no (0.10 of whole lists at 64) | yes (whole lists at 64 on 3 seeds: 1.00, 0.76, 1.00) |
| double | yes | yes (0.93 of whole lists at 64) | not run |
| keep | mostly (0.75 of whole lists) | no | no |
| count | yes (0.96) | no | no |
| twice | no (0.01 of whole lists) | no | no, but two thirds of the items at 64 |

### What the traces show

**Double is a real two-vop loop.** After five ticks of setting up, vops 47 and 43 take turns
for the rest of the run: the write pointer steps every tick and the read pointer every
other tick. Length 12, run with whole steps (the first pointer row is the read pointer that
matters; `>` step forward, `.` stay):

    vop:     24 107  42  42  30  47  47  43  47  43  47  43  47  43  47  43 ...
    read 2:  i0 o23  i0  i0  i0  i1  i2  i2  i3  i3  i4  i4  i5  i5  i6  i6 ...
    write:   o1  o0  o0  o1  o2  o3  o4  o5  o6  o7  o8  o9 o10 o11 o12 o13 ...

So the pc can hold a count of two. The 7% of whole lists it misses at 64 were not looked
into. Bank: 5% of the vops used, 0% of ticks differ between inputs (one fixed program).

**Shift has the loop and a bad start.** One vop (50) runs every tick with the read pointer
one place ahead of the write pointer, which is the routine. But in its first two ticks
the read pointers land on places that differ from list to list (i1 then i2 on one, i7
then i8 on another), which only a move by content can do. As trained that move is a blur
over many slots and the result comes out right; run with whole steps it lands on one
slot, and the loop then starts from the wrong place on most lists.

**Count looks like averaging, not counting.** As trained, the pointers are 0.54 exact,
and on the first tick the likeliest move of all three pointers is seek. A seek with
nothing to match spreads a pointer over the input slots, a load through it is then an
average of the items, and how many are of the kind is a linear function of that average.
Run with whole steps the average is gone and so is the answer (0.96 to 0.02 at the trained
length). The spread itself was not measured; this is read off the moves and the scores.

**Keep never makes the write pointer wait.** The write pointer steps on every tick whatever
the item is (trace: `<` on every tick); the right items land in the right places at trained
lengths by other means that were not untangled (pointers 0.62 exact, a second read pointer
seeking on every tick).

**Twice finds no program**: nine different vops a run, pointers 0.54 exact, seeking on most
ticks. The GRU of the same weights does not learn it in 3,000 steps either (0.35).

### What held it back: the move by content, used as a blur

Seek while training is `softmax(12 x likeness)` over all slots. When nothing matches well,
that is a pointer spread over the whole list, and the load through it is an average of the
list. This is a cheap, smooth thing for the gradient to find, it is right at trained lengths,
and it does not exist when the machine runs with whole steps. Shift and count both used it.

Holding it off needs no change to the core: `tape.no_seek(core)` sets, for every pointer in
every vop, the bias of the wish to seek to -20 and its weights to zero, and zeroes their
gradients. The machine is the same; it never wants that move. With it off:

- shift: carries on all three seeds; two are exact at every length run with whole steps
  and the third gets 0.76 of whole lists at 64 (table above). Seed 0 is one clean program.
  Whole steps, length 6: `vop: 24 49 50 50 50 ...`, both read pointers `in0 > > > ...`,
  write `ou0 ou0 > > > ...`. Seed 1, the inexact one, wrote a different program: it goes
  to the far ends and copies backward, with three ticks of false start.
- count: still not a routine. With the blur gone it reads the whole list first, then
  writes marks at one every two ticks and stops; how many it writes can then only be held
  in the pc, as a size and not as a place (read off the trace, not measured). Right at 8 as trained (1.00 / 1.00), 0.21 with whole steps, 0.00
  at 64. It did not find "step the write pointer when the item is of the kind".
- keep: worse at trained lengths (0.54 of whole lists, from 0.75), and now nearly the same
  with whole steps (0.48), so what it does is exact; it is just not the task. The write
  pointer still steps on every tick. In the traced lists it copies the list with the
  unwanted items left out as holes, which is right only while the wanted items come in an
  unbroken run.
- twice: most of a routine, and not the one I expected (find the mark, go back). It copies the list
  backward into the back half of the output; then a read pointer goes to the last output
  slot and the machine copies the back half of the output into the front half. At length
  12 the traced list is right. The turn from the first copy to the second is not timed
  exactly: 0.28 of whole lists at 8, 0.08 at 16, and 0.67 of items at 64 (0.13 with seek
  on).
- parity (run only with seek off, as one rung above where the climb stopped): 0.95 at 8
  as trained with pointers 0.50 exact, chance at every longer length, chance with whole
  steps. Not a routine. Both GRUs learn it and carry it exactly to 64: a GRU's state can
  be the two-state machine that parity is. The machine would have to change its pc, or
  flip a slot, at every one, and does not find that.

## B. One conditional in isolation

`cond`: input slot 0 is a signal (M copy, R reverse), then the list. `condflash`: the same,
but the signal slot goes dark after the first tick, so nothing in memory says which it was
unless the machine kept it.

| task | seed | model | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| cond | 0 | vvm, as trained | 1.00 / 1.00 | 1.00 / 1.00 | 0.86 / 0.44 | 0.68 / 0.01 |
| | | vvm, whole steps | 0.99 / 0.96 | 0.94 / 0.70 | 0.75 / 0.41 | 0.68 / 0.45 |
| | 1 | vvm, as trained | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| | | vvm, whole steps | 1.00 / 1.00 | 0.97 / 0.92 | 0.96 / 0.94 | 0.96 / 0.95 |
| | 2 | vvm, as trained | 1.00 / 1.00 | 1.00 / 1.00 | 0.95 / 0.63 | 0.73 / 0.04 |
| | | vvm, whole steps | 0.94 / 0.67 | 0.77 / 0.53 | 0.67 / 0.51 | 0.60 / 0.49 |
| | 0 | GRU, same weights | 0.98 / 0.86 | 0.36 / 0.00 | 0.22 / 0.00 | 0.18 / 0.00 |
| | 0 | GRU, same arithmetic | 0.89 / 0.40 | 0.33 / 0.00 | 0.21 / 0.00 | 0.17 / 0.00 |
| cond, seek off | 0 | vvm, as trained | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.75 | 0.72 / 0.00 |
| | | vvm, whole steps | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| condflash | 0 | vvm, as trained | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.83 |
| | | vvm, whole steps | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| | 1 | vvm, as trained | 1.00 / 1.00 | 1.00 / 0.92 | 0.99 / 0.63 | 0.85 / 0.47 |
| | | vvm, whole steps | 0.96 / 0.65 | 0.98 / 0.62 | 0.99 / 0.56 | 0.99 / 0.52 |
| | 2 | vvm, as trained | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.75 | 0.99 / 0.47 |
| | | vvm, whole steps | 0.98 / 0.83 | 0.98 / 0.68 | 0.99 / 0.63 | 0.99 / 0.57 |

(The GRUs read only the items, so the dark signal changes nothing for them; their rows
stand for both tasks.)

**The conditional is learned, in 3,000 steps, on every seed, with no help.** Nothing had to
be found to make it learnable.

**Does the pc change with the signal? Only when it has to.**

- With the signal left lit (`cond`), seeds 0 and 2 do not change the pc. One read pointer
  parks on the signal slot for the whole run, the same vop runs every tick for both
  signals, and that vop steps the write pointer forward from the first output slot when
  the parked slot says copy and backward from the last when it says reverse. The decision
  is a pointer move that depends on a loaded slot, inside one vop. Seed 0, whole steps:

      copy     vop:    48  71  55  55  55  55  55  55 ...    write: o5  o0  o1  o2  o3  o4  o5
      reverse  vop:    48  71  55  55  55  55  55  55 ...    write: o5  o5  o4  o3  o2  o1  o0
      both     read 1: i0  i0  i0  i0 ...  (the signal slot)   read 2: i0 i1 i2 i3 ...

  Seed 1 also parks a read pointer on the signal but runs a different vop for each (82
  for copy, 88 for reverse).
- With the signal left lit and seek held off (`cond 3000 seek=0`, seed 0), no pointer
  parks on the signal. Both read pointers walk the list, the pc takes the signal in the
  first ticks, and each signal gets its own loop (copy `48 48 48 48 49 55 55 ...`,
  reverse `48 48 99 20 20 ...`). This one is exact at every length run with whole steps.
- With the signal dark after the first tick (`condflash`), the pc carries it. The first
  two vops are the same for both signals, and from the third tick the two runs are in
  different loops. Seed 0, whole steps, length 12:

      copy     vop: 24  46   5  52  52  52  52 ...   read: i12 i0 i1 i2 ...   write: o11 o11 o0  o1  o2 ...
      reverse  vop: 24  46  55  28 110 110 110 ...   read: i12 i0 i1 i2 ...   write: o0  o0  o11 o10 o9 ...

  The signal can only have been loaded on the first tick, and the pointers are on the
  dark slot or the list after that, so the pc held it across two ticks and then chose a
  loop. That is a conditional jump. All three seeds do it this way (two loops, one for
  each signal: vops 52 and 110, 51 and 106, 127 and 109), and on all three the choice
  carries: run with whole steps at 64, 0.99 to 1.00 of items are right (a wrong choice
  would leave about half the lists with most items wrong). Seed 0 is exact. Seeds 1 and 2
  get one item wrong in about half the lists at every length, 8 included (in the seed 1
  trace it is the first item of a copied list): a fault in the start of one loop that the
  blur hides at trained lengths, not in the jump.

**Where `cond` is not exact with whole steps (seeds 0 and 2: about half of whole lists at
64, items 0.60 to 0.68), I did not find the cause.** It is more than one item a list. In
the traces at lengths 6 and 12 both branches leave their loop for other vops some ticks
after the list is done, and in the seed 2 trace the reverse branch leaves at about the
moment it finishes; if the same happens before the end of a long list, that is the
failure. Not measured. With seek held off the same seed 0 run is exact at 64 (table), so
here too the moves by content stand between the trained machine and an exact one.

So the owner's guess, that conditional jumps are the weak spot, is not what these runs
show. A jump on a signal is learned easily and carries. What is not learned is a decision
at every item that changes a pointer's step (keep, count), and finding the end of a list
and starting again (twice).

### Bank use and pointer exactness, every run

Bank numbers are over the test runs, as trained and with whole steps together; "differ" is
the share of ticks at which a run's best vop is not the one most runs have at that tick.
Exact is the share of a pointer on its likeliest slot, as trained, at length 6.

| run | pointers exact | vops used | entropy, bits | top vop | ticks that differ |
|---|---|---|---|---|---|
| shift | 0.85 | 3% | 0.36 | 95% | 0% |
| double | 0.81 | 5% | 1.30 | 48% | 0% |
| keep | 0.62 | 53% | 2.33 | 37% | 37% |
| count | 0.54 | 55% | 0.69 | 87% | 13% |
| twice | 0.54 | 29% | 2.24 | 44% | 5% |
| shift, seek off, seeds 0, 1, 2 | 0.91, 0.62, 0.75 | 2%, 79%, 5% | 0.24, 2.01, 0.45 | 97%, 64%, 93% | 0%, 2%, 5% |
| keep, seek off | 0.76 | 25% | 2.23 | 35% | 41% |
| count, seek off | 0.55 | 55% | 2.12 | 35% | 20% |
| twice, seek off | 0.76 | 45% | 2.68 | 29% | 46% |
| parity, seek off | 0.50 | 58% | 1.67 | 67% | 26% |
| cond, seeds 0, 1, 2 | 0.77, 0.87, 0.86 | 43%, 6%, 7% | 2.49, 1.52, 1.90 | 39%, 51%, 50% | 30%, 44%, 23% |
| cond, seek off | 0.89 | 4% | 1.36 | 47% | 43% |
| condflash, seeds 0, 1, 2 | 0.73, 0.67, 0.69 | 5%, 7%, 4% | 1.51, 1.38, 1.22 | 48%, 50%, 61% | 44%, 44%, 23% |

The working routines use a handful of vops (2% to 7% of the bank). The runs that use half
the bank are the ones that found no routine; there, more vops in use goes with failing,
not with a harder task being met. No pointer is ever fully exact as trained, even in the
runs that are exact with whole steps.

## C. The lights game

Not done here: moved to another session before any run. Left in `vvm2/tape.py` for it,
none of them run beyond a check that the file loads:

- `clock=1`: one more input slot, a 4-bit counter of the tick number that wraps at 16
  (`clock_code`); the GRU sees it too.
- `hold=N` was already there and already has no gap between lights.
- `lights=N`: train on 1 to N lights (6 unless said).
- `seek=0`: hold the moves by content off (`tape.no_seek`), for copy, reverse and simon.
- `threads=N`.

One thing worked out on paper that may save that session a run: at one tick a light the
first button is read on the same tick the play signal first shows, so the machine cannot
wait for the signal before moving its write pointer to the button. There is still an exact
routine: write the first light straight to the button slot, the rest to scratch; when the
signal changes, send the write pointer to the button and a read pointer to the first
scratch slot, and step.

## What is missing from the machine (nothing here was changed)

1. **A task cannot ask for exact pointers.** `core.exact` is detached and `core.trace`
   holds detached copies, so no loss can be put on pointer blur from outside. A task can
   only hope the blur goes away. Shift and count are learned through the blur.
2. **A task cannot leave a move out.** Seek is always among a pointer's choices. Holding
   it off took reaching into the weights and their gradients (`tape.no_seek`). A switch on
   the core, or a seek that is exact while training (the gradient passed straight through),
   would be the clean form.
3. **Nothing tells the machine a list has ended.** A step past the last slot stays there
   and the last item is loaded again, looking like a new item. Every task that must stop
   or turn round at the end (twice, count, parity, compare, adding, sort) needs a mark
   from the world, and then a jump on it.
4. **Nothing makes a loop's pc stay put.** A loop is a vop whose next pc fetches itself
   again; over many ticks the pc may creep to another vop. The `cond` traces show loops
   being left some ticks after the list is done; whether that is why two seeds fail at 64
   was not measured.
5. **The machine cannot stop or ask for more ticks.** After the answer is written it keeps
   running, and anything it does then can spoil the answer (in `cond` the wandering after
   the loop is harmless only because the write pointer ends up on an input slot).

## What is suspect

- Every ladder row is one seed and 3,000 steps, except shift with seek off (3 seeds).
  `cond` with seek off is one seed.
- "Whole answers right" at 64 on 300 lists; 0.93 for double means about 20 lists wrong
  and I did not look at which.
- The count and keep failures may be undertraining: lookup needed 16,000 steps where
  4,000 failed (finding 36). No longer run was made.
- Keep and count split the 8 symbols into two kinds by number (0 to 3, 4 to 7), which the
  machine has to learn from scratch as a rule on 16 numbers. A kind that is easier to see
  was not tried.
- The trace shows only the best-matching vop of the 4 fetched. "The same vop for both
  signals" means the same best match; the blend behind it may differ.
- The runs were made while the CPU was heavily shared, some at 4 threads and some at 1,
  some on the old tick and some on the faster one. Scores should not depend on that; it
  was not checked run by run.
- The seed 1 and 2 runs and the `seek=0` runs (parity apart) have no GRU beside them
  (`gru=0`, or stopped once the machine's rows were written): the GRU does not depend on
  the machine's seek, and seed 0 has it. `condflash` has no GRU row of its own; the GRU
  sees the same items as in `cond`.

## Not done

- Rungs above twice: compare, adding (binary, decimal), brackets, sort, and parity with
  seek on. The tasks are written and run with `--quick`; no scoring run of them was made.
- Double with seek off, and double on more than one seed.
- Three seeds for anything on the ladder except where the tables say.
- Longer training, a smaller bank, pointers started on chosen slots (`boot=1`), the
  seek-my-twin step (`twin=1`): all are options, none was needed for B and none was tried
  on A.
- A small transformer beside the machine.
- The lights game (part C).

## Script lines

    scripts/algos.sh shift 3000            # and double, keep, count, twice, cond, condflash
    scripts/algos.sh cond 3000 seed=1      # and seed=2
    scripts/algos.sh condflash 3000 seed=1 gru=0       # and seed=2
    scripts/algos.sh shift 3000 seek=0     # and seed=1 gru=0, seed=2 gru=0
    scripts/algos.sh count 3000 seek=0 gru=0           # and keep, twice, cond
    scripts/algos.sh parity 3000 seek=0
    scripts/algos.sh --quick               # every task for 30 steps

Output is kept in `out/algos/` (not tracked).

## Proposed edits

For `docs/findings.md`, after 37:

> 38. **A conditional jump is learned easily; the weak spot is elsewhere.** [likely: 3 seeds,
>     `scripts/algos.sh cond 3000`, `... condflash 3000`] One signal slot says copy or
>     reverse. With the signal left lit the machine usually does not jump at all: it parks a
>     read pointer on the signal and one vop steps the write pointer forward or backward by
>     what that slot holds (2 of 3 seeds; right at trained lengths on all 3; with whole
>     steps one seed carries to 64 and two lose about half the lists there). With the signal
>     dark after the first tick it holds the signal in the pc for two ticks and then runs a
>     different loop for each, on all 3 seeds; run with
>     whole steps at 64 the choice is right on all 3 (items 0.99 to 1.00), one seed is exact
>     and two get one item wrong in about half the lists.
> 39. **While training, the move by content is a blur over the list, and the machine uses
>     the blur.** [likely for shift: 3 seeds with seek off, 1 with it on; hint for the rest;
>     `scripts/algos.sh shift 3000`, `... seek=0`] Shifting
>     a list by one place is learned as the right loop with a start that seeks, right as
>     trained and 0.23 with whole steps at the trained length. Counting items of one kind is
>     learned by spreading a pointer over the list and reading the average (1.00 / 0.96 as
>     trained, 0.02 with whole steps). With seek held off from outside, shift carries
>     on 3 seeds (whole lists at 64: 1.00, 0.76, 1.00), and the copy-or-reverse conditional
>     goes from half the lists at 64 to exact (one seed). Finding 35 said copy and reverse were unharmed by the extra
>     move; shift, one place different from copy, is harmed.
> 40. **The ladder of small algorithms stops early.** [hint: one seed, 3,000 steps,
>     `scripts/algos.sh`] Trained on 2 to 8, run with whole steps at 64: writing every item
>     twice carries (a loop of two vops; 0.93 of whole lists), shift carries only with seek
>     off, and keeping the items of one kind, tallying them, and writing a list twice over
>     do not become routines. What the failures share is a pointer that must step or not by
>     the item just loaded, or a turn at the end of the list. No GRU carries any of those.
>     Parity (seek off) is at chance past the trained length, and there both GRUs are exact
>     at 64.

For `docs/ideas.md`:

> - **Seek that is exact while it learns:** take the best match outright and pass the
>   gradient straight through, so a pointer is never spread over the list and nothing can
>   be learned from an average of slots.
> - **A loss on pointer blur from outside:** let `core.exact` keep its gradient (or return
>   the pointers), so a task can ask for single-slot pointers.
> - **A move switch per task:** a list of the moves a device allows, so a line of items can
>   be given to the machine with no move by content.
> - **An end the machine can feel:** a step that cannot be taken (already at the last slot)
>   shows up in what the vop loads, so a loop can know it has run out without a mark from
>   the world.
> - **Holding the pc:** a vop can say "same pc again" as a choice (as a pointer can say
>   stay), so a loop cannot creep.

## The predictions, and how they did

Written and committed (commit a564ee9) before any scoring run.

| # | task | predicted: learned at trained lengths / carries with whole steps | what happened |
|---|---|---|---|
| 1 | shift | yes / yes | yes / no (yes with seek off) |
| 2 | double | yes / yes, 70% sure | yes / yes |
| 3 | keep | yes / yes, 60% sure | mostly / no |
| 4 | count | yes / even odds | yes / no |
| 5 | twice | mostly (items above 0.9) / no | no (items 0.83) / no |
| 6 | parity | no / no | seek off only: 0.95 at 8 as trained (I said under 0.7) / no; the GRU carries it, as I half expected |
| 7 to 11 | compare, add, sort, brackets | partly / no | not run |

B: predicted learned at trained lengths, carrying on at least one seed of three, 40% that
it would keep a read pointer on the signal and use the same vops for both: that is what two
seeds of three did. `condflash`: predicted even odds of being learned and one in three of
carrying; it was learned on all three seeds, the choice carries
on all three, and one seed is exact at 64.

Wrong in kind: I expected the one-vop tasks (shift, keep, count) to be easy and the
conditional jump to be the hard part. It was the other way round. The reasoning "a vop's
pointer steps are a linear function of the item it loaded, so stepping or not by the item
needs no jump" says the program exists, not that training finds it.

C was not run; its predictions are in the first commit of this file.
