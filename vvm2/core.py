"""The vector virtual machine.

Memory is one list of slots. Every slot is the same: 64 numbers, read as an 8 by 8 grid.
A slot can be data, an op (a vop), or the home of a core's addresses. Nothing marks which.

An address is one number from 0 to 1, spread evenly over memory: 0 is the first slot, 1 the
last, and it lands on the one slot it is nearest.

A core has three addresses: its pc, its load address and its store address. They are the
first three numbers of a slot that belongs to the core, so they are in memory like
everything else and any core can write them.

One tick, every core at once:
    the pc lands on a slot           that slot is the op
    the load address lands on a slot that slot is loaded
    the op is matmulled into it      an 8 by 8 grid times an 8 by 8 grid, each number then
                                     held between -1 and 1
    the store address lands on a slot  the result replaces what was there

A store is a store: the slot becomes the result. There is no loudness and no mixing with
what was there before. Numbers stay between -1 and 1 because every result is held there.

Nothing steps and nothing walks. An address changes only when an op's result is written to
the slot that holds it. All cores read memory as it stood when the tick began; then their
writes land, in core order. The world owns some slots (inputs): it writes them and the
program's writes to them are dropped.

This file is the machine and nothing else: no training, no task.
"""
import numpy as np

SIDE = 8
SLOT = SIDE * SIDE


def land(address, n):
    """An address (0 to 1) -> the one slot it is nearest, of n."""
    return int(round(float(np.clip(address, 0.0, 1.0)) * (n - 1)))


def address_of(slot, n):
    """The address that lands exactly on a slot."""
    return slot / (n - 1)


class Machine:
    def __init__(self, memory, cores, world=()):
        """memory  [n, 64], what memory holds when the run starts
        cores   the slots that hold each core's three addresses
        world   the slots the world owns (the program cannot write them)"""
        self.memory = np.array(memory, dtype=np.float32)
        self.cores, self.world = list(cores), set(world)

    def tick(self):
        """One tick. Returns, for each core: the slots its pc, load and store addresses
        landed on, and the result it wrote."""
        mem, n, done = self.memory, len(self.memory), []
        for home in self.cores:
            pc, load, store = (land(a, n) for a in mem[home, :3])
            result = np.clip(mem[pc].reshape(SIDE, SIDE) @ mem[load].reshape(SIDE, SIDE), -1, 1)
            done.append((pc, load, store, result.ravel()))
        for pc, load, store, result in done:
            if store not in self.world:
                mem[store] = result
        return done
