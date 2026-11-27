// The vector virtual machine, for speed. The same machine as vvm2/core.py.
//
// Memory is one list of slots; every slot is 64 numbers, read as an 8 by 8 grid. An address
// is a number from 0 to 1 spread evenly over memory; it lands on the one slot it is nearest.
//
// A vop is an op slot and the slot right after it, which holds the vop's three addresses in
// its first three numbers: where to load from, where to store, and which vop comes next.
// A core is a pc: the first number of a slot that belongs to the core.
//
// One tick, every core: the pc lands on a slot; that slot is the op and the next slot has
// its addresses. The slot under the load address is loaded, the op is matmulled into it
// (each number then held between -1 and 1), the result replaces the slot under the store
// address, and the core's pc becomes the vop's "next". Nothing steps. All cores read memory
// as it stood when the tick began; then their stores land, in core order, then the pcs are
// set. Stores to the world's slots are dropped.
#pragma once
#include <cmath>
#include <cstring>
#include <vector>

static const int SIDE = 8, SLOT = 64;

inline int land(float a, int n) {
    a = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
    return (int)std::lrintf(a * (float)(n - 1));
}
inline float address_of(int slot, int n) { return (float)slot / (float)(n - 1); }

struct Machine {
    int n = 0;                    // slots
    std::vector<float> mem;       // n * 64
    std::vector<int> cores;       // the slots whose first number is a core's pc
    int world_lo = 0, world_hi = 0;   // the world owns slots [world_lo, world_hi)
    std::vector<float> results, nexts;   // work space
    std::vector<int> stores;

    float* slot(int i) { return mem.data() + (size_t)i * SLOT; }

    void tick() {
        const size_t c = cores.size();
        results.resize(c * SLOT);
        stores.resize(c);
        nexts.resize(c);
        for (size_t k = 0; k < c; k++) {
            int p = land(slot(cores[k])[0], n);
            if (p > n - 2) p = n - 2;                 // a vop needs the slot after it
            const float* op = slot(p);
            const float* ad = slot(p + 1);
            const float* x = slot(land(ad[0], n));
            stores[k] = land(ad[1], n);
            nexts[k] = ad[2];
            float* out = results.data() + k * SLOT;
            for (int i = 0; i < SIDE; i++) {
                float row[SIDE];
                for (int q = 0; q < SIDE; q++) row[q] = 0.0f;
                for (int j = 0; j < SIDE; j++) {
                    const float a = op[i * SIDE + j];
                    for (int q = 0; q < SIDE; q++) row[q] += a * x[j * SIDE + q];
                }
                for (int q = 0; q < SIDE; q++)
                    out[i * SIDE + q] = row[q] > 1.0f ? 1.0f : (row[q] < -1.0f ? -1.0f : row[q]);
            }
        }
        for (size_t k = 0; k < c; k++)
            if (stores[k] < world_lo || stores[k] >= world_hi)
                std::memcpy(slot(stores[k]), results.data() + k * SLOT, SLOT * sizeof(float));
        for (size_t k = 0; k < c; k++) slot(cores[k])[0] = nexts[k];
    }
};
