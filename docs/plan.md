# Plan

Three claims, to be earned in this order. Each depends on the one before. Nearly every
finding so far is one seed; a claim is not earned until it holds on several.

## 1. It computes like a machine

It learns routines that carry beyond what it was trained on.

- *(done, 3 seeds)* Copy and reverse: trained on lengths 2 to 8, exact at 96 (findings 33, 37).
- *(done, 1 seed)* Lookup by key: trained on 2 to 6 pairs, right at 24 (finding 36).
- *(open)* The lights game: learned at trained lengths only, not as a routine (finding 34).

**The battery: find where the limits are.** Each is its own task file on the one core, with
a plain net beside it, tested beyond what it trained on, run with whole steps, with a trace.

- *(running)* Images through pointers that move on a grid and zoom between a coarse and a
  fine copy: MNIST, Kuzushiji, Fashion; circles, squares and triangles of any size and
  place, named and then drawn; generating clothes and bags, later dogs (needs colour and
  another data set).
- *(running)* Cellular automata: one-line rules, then Conway's, learned on small boards and
  run on larger ones.
- *(running)* Algorithms and conditionals: adding with a carry, counting, parity, sorting,
  matching brackets, keeping only some items; the lights game at long holds, with and
  without a clock device.
- *(done)* Ordinary curves and a pole (findings 38, 39). To fix: the world cannot see the
  output slots while the machine runs (`world(t)` is not handed them), so the pole task
  reaches into the running core; the core should hand them over. Not yet run: the pole
  with velocities hidden, swing-up.
- *(next, after the image device is in)* Cats against dogs, then all ten classes of a
  small colour data set (CIFAR-10): a colour patch a slot, coarse and fine levels. Then look
  inside: for each vop, what is under the pointer when it runs, and where it sends the
  pointer next. The owner's question: is there a vop for fur, one for an ear, one that means
  "look to the left of the right eye".
- Later: walking, picking things up; an image seen through a window it moves.

## 2. A big bank is cheap

Most of the bank sits unread, and a bigger one still helps at the same compute. Seen once,
with small vops in an earlier form (finding 4). To show on this machine: bank size against
score at fixed compute on a task with many sub-skills; how much of the bank is used;
noise in each place (pc, addresses, stored values, vops) and what it does to that; fetch by
a bar; growing a vop when nothing matches. Then the engine and the disk.

## 3. It can change itself, and it never stops

- **A machine on its own clock, never reset** (owner): it ticks at a fixed rate whether or
  not anything arrives. The world writes inputs and reads outputs when it likes, swaps
  devices in and out, and changes the task, with nothing wiped. Then time itself is
  something it can learn: how long ago, how long between. Trained on a stream in windows,
  with memory, pointers and the pc carried from one window to the next; this is where
  credit past the window matters (ideas 24).
- **A new device on a running machine** (owner's bet): no special shake-up is needed; give
  it tasks that can only be done through the new slots and it will learn that part of
  memory. To compare with: a short burst of noise or curiosity aimed at the new slots.
- Vops in the same memory, written while running.

## Alongside

- More seeds and a baseline beside everything.
- Smaller vops: the best image machine has 5 million weights and uses a tenth of them.
- Exploration driven by the machine's own state (curiosity), not by a timer.
