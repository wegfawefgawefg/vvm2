"""The vector virtual machine. (cpp/vvm.hpp is the same machine, for speed.)

Memory is one list of slots. Every slot is the same: 64 numbers, read as an 8 by 8 grid.
A slot can be data, an op, or hold addresses. Nothing marks which.

An address is one number from 0 to 1, spread evenly over memory: 0 is the first slot, 1 the
last, and it lands on the one slot it is nearest.

A vop is an op slot and the slot right after it. The first three numbers of that second
slot are the vop's addresses: where to load from, where to store, and which vop comes next.

A core is a pc: the first number of a slot that belongs to the core.

One tick, every core at once:
    the pc lands on a slot              that slot is the op; the slot after it has the
                                        vop's three addresses
    the load address lands on a slot    that slot is loaded
    the op is matmulled into it         an 8 by 8 grid times an 8 by 8 grid, each number
                                        then held between -1 and 1
    the store address lands on a slot   the result replaces what was there
    the core's pc becomes the vop's "next"

A store is a store: the slot becomes the result. Nothing steps and nothing walks. A vop's
addresses sit in an ordinary slot, so another vop can load them, change them and store them
back. All cores read memory as it stood when the tick began; then their stores land, in
core order; then the pcs are set. The world owns some slots (inputs): it writes them and
the program's stores to them are dropped.

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
        cores   the slots whose first number is a core's pc
        world   the slots the world owns (the program cannot write them)"""
        self.memory = np.array(memory, dtype=np.float32)
        self.cores, self.world = list(cores), set(world)

    def tick(self):
        """One tick. Returns, for each core: the slot its pc landed on, the slots its vop
        loaded from and stored to, the address of the next vop, and the result."""
        mem, n, done = self.memory, len(self.memory), []
        for home in self.cores:
            pc = min(land(mem[home, 0], n), n - 2)          # a vop needs the slot after it
            load, store, nxt = land(mem[pc + 1, 0], n), land(mem[pc + 1, 1], n), mem[pc + 1, 2]
            result = np.clip(mem[pc].reshape(SIDE, SIDE) @ mem[load].reshape(SIDE, SIDE), -1, 1)
            done.append((pc, load, store, float(nxt), result.ravel()))
        for pc, load, store, nxt, result in done:
            if store not in self.world:
                mem[store] = result
        for home, (pc, load, store, nxt, result) in zip(self.cores, done):
            mem[home, 0] = nxt
        return done
