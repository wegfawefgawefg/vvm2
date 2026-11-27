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
