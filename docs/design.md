# The design

What the machine is now. The code is `vvm2/core.py`, and its header says the same thing in
fewer words. How it got here, and the five earlier forms, are at the end.

## The machine

A virtual machine in which every value is a vector and every instruction is a learned op (a
vop). It is fixed and knows nothing: it is the same for every task. What it does is decided
by what is in its memory and its bank.

**Memory** is one list of slots, 16 numbers each, in regions:

- slot 0, the pc. Ordinary memory.
- input: written by the world. The program can read it and cannot write it.
- scratch: starts empty.
- output: written by the program, read by the world.

**Pointers.** The machine has a few read pointers and one write pointer. A pointer is a
place: exactly one slot. A vop moves it in one of two ways:

- by position: stay; take one of the device's own steps; or go to the first or last slot of
  a region. The machine has a fixed number of step kinds and does not know what they mean.
  The device says where each leads from each of its slots: next and previous on a line;
  right, left, down, up on a grid; zooming between a coarse and a fine copy of an image.
- by content ("seek"): go to the slot whose contents best match a vector the vop supplies.
  An optional built-in form is "go to another slot like the one I am on".

These are two different basic operations. Stepping knows about order and nothing about
contents; seeking knows about contents and nothing about order. Each can only fake the other
badly (finding 32 is what happened when content addressing was made to stand in for
position).

**A vop** is a key and one matrix with a bias. The bank of vops is separate from memory for
now.

**One tick:**

1. Fetch: the few vops whose keys are nearest the pc slot, blended by how near. This is the
   only thing that chooses what runs.
2. Load: the slots under the read pointers and under the write pointer. Always.
3. Execute: one matmul. In: what was loaded and the pc slot. Out: a value to store, the next
   pc, and for every pointer how much it wants each move and what to seek.
4. Store: the value goes to the slot under the write pointer. Always; a silent value changes
   nothing. The pc slot takes its next value; the pointers move.

**One write rule** for every store: a slot moves toward what is written by its loudness,
`slot = (1 - |c|) slot + c` with `|c|` at most 1.

**Soft while learning, exact when running.** The only softness is in the fetch. Because a
few vops are blended, a pointer's move is a blend of moves while the machine learns, and the
pointer is spread over a few slots. Run "with whole steps" (`core.hard = True`), every
pointer takes its likeliest move outright and is always one slot. A routine that is real
works at any length that way (finding 33).

**Start and stop.** The pc starts from a learned reset value and each pointer from a learned
choice of region end, the same for every run. The world decides how many ticks to run and
when to read the output.

**What is not in it:** a dense layer, an input map, a readout, address vectors on slots, any
state outside memory and the pointers.

## Why each part is as it is (the evidence is in docs/findings.md)

- Pointers are exact places because pointers found by matching address vectors confused
  places that looked alike and blurred on long lists (findings 32, 33).
- Both kinds of move because lists need stepping and lookup needs seeking (findings 33, 36).
- The vop sets the next pc and the pointer moves itself, in one matmul, because the fetch is
  the expensive part of a tick and more should be done per fetch (finding 20).
- Soft fetch because an exact choice gives the gradient nothing to follow (finding 26).
- The clamp because it is as cheap as a compare; the write rule because it bounds every
  vector without a second mechanism.

## Not built

- One memory for vops too, so the program can write its own instructions.
- A slot that can hold a place, so a pointer can be saved and loaded (indirect addressing).
- Fetch by a bar in place of a fixed few, and growing a vop when nothing matches.
- Halting from inside (an early form existed in version 2: old/halt.py).
- Drivers with control slots (where to look, how far to zoom).
- Anything of the engine: fetch by hash, 8-bit numbers, grouping runs by vop, fetching
  ahead, the bank on disk.

## The earlier forms (code in old/)

1. Ops as vectors added to the state, plus a dense layer (old/vvm.py, old/bank.py). Not
   the idea. Its product-key lookup is a real way to address a very large bank.
2. Ops as full matrices on a state that was the input (old/vop.py). One op did everything
   and the bank sat idle. The tree pictures, paths per digit and first halting are from it.
3. Input and output as their own regions, flat vops reading all of memory (old/mm.py). The
   best score for its size and most of the bank in use; no addresses at all.
4. Memory as vectors, vops with built-in slot numbers (old/ram.py). Addresses were wired
   in, not computed.
5. A pc and address lanes, with the state apart from memory (old/vm.py).
6. One memory with the pc and address slots in it, pointers as address vectors matched by
   nearness (old/core6.py, then old/core6b.py with any input and output slots). Reached
   0.986 to 0.989 on MNIST; on lists it confused places whose addresses looked alike.
7. This one: exact pointers moved by steps and by seeking.
