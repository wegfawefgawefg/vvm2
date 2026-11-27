# Working here

vvm2 is a research repo: a neural net shaped like a computer (registers, a learned program
bank of ops read sparsely and applied to the state, a run-time data memory, and no dense
layer on the state), built so the bank can live
on disk. Read `README.md`, `docs/design.md` and `docs/plan.md` first. The repo is public:
no private paths, names, hostnames or keys in anything tracked.

## How results are made

- Gradient training and ordinary deep-learning practice are welcome. The question is
  whether this shape of machine works, not whether backprop can be avoided.
- Every claim is beside baselines (an MLP, a GRU, a small transformer) matched on parameter
  count and, separately, on compute per step. Report both numbers every time.
- Write your predictions into the results doc and commit them before reading any scoring run.
- Screen on 2 or 3 seeds and short runs first; spend more seeds only on what survives.
- Report bank use on every run: share of entries ever used, entropy of use, the top
  entry's share, and how much the picked entries change with the input. Using little of the
  bank is fine when the score is good. The fault is being locked in: a poor score beside
  unused entries, the same entries whatever the input, or no more of the bank drawn on
  when the task grows or changes.
- Look inside a win before calling it one: what does the vvm do tick by tick.
- A null stated cleanly is a good result. Say what you did not do.

## The core is frozen while tasks are tried

`vvm2/core.py` is the machine, and `docs/design.md` says what it is and why. A task does not
change it. A task is a world around the core: it supplies the input slots, the device's
steps (`go`), the number of output slots and the ticks, and reads the output slots
(`vvm2/tape.py` is the pattern). If a task cannot be done without changing the machine, do
not change it: write down what is missing and why, as a finding. Always report the machine
"run with whole steps" (`core.hard = True`) beside the trained run, and show a trace.

## Layout and style

- `vvm2/`: the Python package (model, baselines, tasks, harness). `scripts/`: one shell
  script per experiment, its recipe in the header; the owner runs scripts, so docs quote
  script lines. `docs/results-YYYY-MM-DD-name.md`: one write-up per experiment. `out/` and
  `data/` are not tracked; scripts fetch data and remake outputs.
- Files under 500 lines. Plain words, no emoji, no time estimates. Small functions, no
  framework beyond torch and numpy unless there is a reason written down.
- Your own script must run end to end with a `--quick` setting before any hand-back.

## Machine manners

- At most 2 jobs at once on this PC, each with 4 threads (`torch.set_num_threads(4)`), always
  `nice -n 19`. Four agents share the CPU.
- Never open a window. Never kill by pattern; exact pids of processes you started only.
- No `rm` on variable globs. If disk is low, report; do not delete to make room.
- Commit in your own worktree; do not push. Do not leave anything running, and do not end
  a turn while your own jobs run: wait for them, then hand back once.
