// The vector virtual machine, the same as vvm2/core.py, for speed.
//
// Memory is one list of slots; every slot is 64 numbers, read as an 8 by 8 grid. An address
// is a number from 0 to 1 spread evenly over memory; it lands on the one slot it is nearest.
// A core is a slot whose first three numbers are its pc, its load address and its store
// address. One tick, every core: the slot under the pc is the op, the slot under the load
// address is loaded, the op is matmulled into it (each number then held between -1 and 1),
// and the result replaces the slot under the store address. Nothing steps. All cores read
// memory as it stood when the tick began; then their stores land, in core order. Stores to
// the world's slots are dropped.
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
    std::vector<int> cores;       // the slots that hold each core's three addresses
    int world_lo = 0, world_hi = 0;   // the world owns slots [world_lo, world_hi)
    std::vector<float> results;   // work space: one result a core
    std::vector<int> stores;

    float* slot(int i) { return mem.data() + (size_t)i * SLOT; }

    void tick() {
        const size_t c = cores.size();
        results.resize(c * SLOT);
        stores.resize(c);
        for (size_t k = 0; k < c; k++) {
            const float* home = slot(cores[k]);
            const float* op = slot(land(home[0], n));
            const float* x = slot(land(home[1], n));
            stores[k] = land(home[2], n);
            float* out = results.data() + k * SLOT;
            for (int i = 0; i < SIDE; i++) {
                float row[SIDE] = {0, 0, 0, 0, 0, 0, 0, 0};
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
    }
};
