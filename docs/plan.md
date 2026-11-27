# Plan

Three claims, to be earned in this order. Each depends on the one before. Nearly every
finding so far is one seed; a claim is not earned until it holds on several.

## 1. It computes like a machine

It learns routines that carry beyond what it was trained on.

- *(running)* Lists on a line of memory: copy, reverse. Trained on lengths 2 to 8, tested
  on 8 to 24. `scripts/tape.sh copy`.
- *(running)* The lights game (Simon): the sequence arrives one light at a time and has to
  be stored and played back. `scripts/tape.sh simon`.
- Adding two numbers digit by digit with the carry.
- Moved digits, where a first sign of a shared routine was seen (finding 28): the same
  test on the new core, then shrunk, enlarged and turned digits.
- An image seen through a small window the machine moves and zooms.

## 2. A big bank is cheap

Most of the bank sits unread, and a bigger one still helps at the same compute. Seen once,
with small vops in an earlier form (finding 4). To show on this machine: bank size against
score at fixed compute on a task with many sub-skills; how much of the bank is used;
noise in each place (pc, addresses, stored values, vops) and what it does to that; fetch by
a bar; growing a vop when nothing matches. Then the engine and the disk.

## 3. It can change itself

Vops in the same memory, written while running; a stream of inputs with nothing wiped; a
new device added to a trained machine; learning without end.

## Alongside

- More seeds and a baseline beside everything.
- Smaller vops: the best image machine has 5 million weights and uses a tenth of them.
- Exploration driven by the machine's own state (curiosity), not by a timer.
