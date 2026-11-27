# The design

What the machine is now. The code is `vvm2/core.py`, and its header says the same thing in
fewer words. How it got here, and the five earlier forms, are at the end.

## The machine

A virtual machine in which every number is a vector, every address is a vector matched by
nearness, and every instruction is a learned op (a vop). It is fixed and knows nothing: it
is the same for every task. What it does is decided by what is in its memory.

**Memory** is one list of slots, 16 numbers each. Every slot has an address, also 16
numbers: 4 for the region, 12 for the position inside the region. Regions:

- head: the machine's own slots. Slot 0 is the pc; then the read addresses; then the write
  address. They are ordinary memory, and anything can point at them.
- input: written by the world. The program can read it and cannot write it.
- scratch: starts empty.
- output: written by the program, read by the world.

The world decides how many input and output slots there are and gives each its position: a
line for a list, a grid for an image, nothing in particular for a single light or button.
Adding a device is giving it a region and positions nobody uses.

**A vop** is a key and one matrix with a bias. The bank of vops is separate from memory for
now (see "not built").

**One tick:**

1. Fetch: the few vops whose keys are nearest the pc slot, blended by how near.
2. Load: the memory at each read address, and at the write address. An address picks the
   slots whose addresses are nearest, weighted by how near (a spotlight, not a point).
3. Execute: one matmul. In: what was loaded, and the head slots as they stand. Out: a value
   to store, the next pc, the next read addresses, the next write address. Clamped.
4. Store: the value goes to the write address; the head slots take their next values.

**One write rule** for every store, in memory or in the head: a slot moves toward what is
written by its loudness, `slot = (1 - |c|) slot + c` with `|c|` at most 1. A quiet vop
nudges, a loud one replaces, a silent one does nothing. No vector in the machine is ever
longer than 1, and nothing is tuned.

**Start and stop.** The head starts from a learned reset value, the same for every run;
scratch and output start empty. So the first tick is the same for every input: a boot
routine. The world decides how many ticks to run and when to read the output.

**What is not in it:** a dense layer, an input map, a readout, vops with built-in slot
numbers, any state outside memory. Only the pc chooses what runs; the input never does.

## Why each part is as it is (the evidence is in docs/findings.md)

- The vop sees the head slots because a machine that cannot see its own pointers cannot
  step them (finding 23).
- Several reads a tick because one 16-number load a tick starves it (finding 22).
- The vop sets the next pc and addresses itself, not by separate stores, because the fetch
  is the expensive part of a tick and more should be done per fetch (finding 20).
- Soft pointers because a pointer that reads exactly one slot cannot be trained
  (finding 26).
- The clamp because it is as cheap as a compare; the write rule because it bounds every
  vector without a second mechanism.

## Not built

- One memory for vops too, so the program can write its own instructions.
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
6. This one. Its first form, with image patches built in, is old/core6.py.
