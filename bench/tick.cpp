// How fast is one tick on this CPU, in the two forms of a vop?
//
//   pure   a vop is a 16 x 32 matrix. A tick: fetch the vop the pc slot points at, load the
//          slot the read address points at and the slot the write address points at, one
//          matmul, one store. Changing the pc or a pointer is a store too, so it costs a
//          whole tick: four pure ticks do what one wide tick does.
//   wide   a vop is a 64 x 32 matrix. A tick: the same fetch and loads, one matmul that
//          gives four vectors: the value to store, the next pc, the next read address and
//          the next write address. Four stores.
//
// Everything is a slot of 16 floats in one memory; every store uses the one write rule
// (a slot moves toward what is written by its loudness). Fetching by address is stood in
// for by a hash of the pc slot, as a hashed fetch would do: what is measured is the cost of
// jumping to an unpredictable vop in a bank of a given size, then doing the arithmetic.
// The next fetch depends on what this tick computed, so ticks of one run cannot overlap;
// with several runs sharing the bank the CPU can overlap their fetches.
//
//   bench/tick <bank megabytes> <runs> <ticks per run>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const int SLOT = 16;

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static inline uint64_t rng() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static inline float unit() { return (float)(rng() >> 40) / 8388608.0f - 1.0f; }  // -1..1

// an address vector -> an index: the stand-in for a hashed fetch. Two bits a number (its
// sign, and whether it is large), so nearby vectors give the same index; no long chain.
static inline uint64_t where(const float* v) {
    uint32_t sign = 0, big = 0;
    for (int i = 0; i < SLOT; i++) {
        sign |= (uint32_t)(v[i] > 0.0f) << i;
        big |= (uint32_t)(std::fabs(v[i]) > 0.25f) << i;
    }
    uint64_t h = ((uint64_t)big << 16 | sign) * 0x9E3779B97F4A7C15ull;
    return h ^ (h >> 31);
}

// out[rows] = clamp(m . in[32]); m is stored column by column (32 columns of `rows`), so
// the inner loop is one vector multiply-add per column
static inline void apply(const float* m, const float* in, float* out, int rows) {
    for (int r = 0; r < rows; r++) out[r] = 0.0f;
    for (int j = 0; j < 32; j++) {
        const float* col = m + j * rows;
        const float x = in[j];
        for (int r = 0; r < rows; r++) out[r] += col[r] * x;
    }
    for (int r = 0; r < rows; r++) out[r] = out[r] > 1.0f ? 1.0f : (out[r] < -1.0f ? -1.0f : out[r]);
}

// the one write rule: slot = (1 - |v|) slot + v, with |v| held to 1 at most
static inline void write(float* slot, const float* v) {
    float n = 0.0f;
    for (int i = 0; i < SLOT; i++) n += v[i] * v[i];
    n = std::sqrt(n);
    float k = n > 1.0f ? 1.0f / n : 1.0f;
    float keep = 1.0f - (n > 1.0f ? 1.0f : n);
    for (int i = 0; i < SLOT; i++) slot[i] = keep * slot[i] + k * v[i];
}

struct Run {            // one run's memory: slot 0 pc, 1 read address, 2 write address, then data
    float* mem;
};
static const int SLOTS = 64;

static double bench(bool wide, size_t bank_bytes, int runs, long ticks) {
    const int rows = wide ? 64 : 16;
    const size_t vop_floats = (size_t)rows * 32;
    const size_t count = bank_bytes / (vop_floats * 4);
    float* bank = (float*)std::aligned_alloc(64, count * vop_floats * 4);
    for (size_t i = 0; i < count * vop_floats; i++) bank[i] = unit() * 0.25f;
    std::vector<Run> rs((size_t)runs);
    for (auto& r : rs) {
        r.mem = (float*)std::aligned_alloc(64, SLOTS * SLOT * 4);
        for (int i = 0; i < SLOTS * SLOT; i++) r.mem[i] = unit() * 0.5f;
    }
    std::vector<size_t> last((size_t)runs, 0);
    long jumps = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (long t = 0; t < ticks; t++) {
        for (size_t ri = 0; ri < rs.size(); ri++) {
            Run& r = rs[ri];
            float* mem = r.mem;
            // pure: the pc moves on by itself every tick (the stand-in for "next"), as a
            // program counter does; wide: the vop wrote the next pc
            size_t at = (where(mem) + (wide ? 0 : (uint64_t)t * 0x9E3779B97F4A7C15ull)) % count;
            jumps += at != last[ri];
            last[ri] = at;
            const float* vop = bank + at * vop_floats;                           // fetch
            float* from = mem + (3 + where(mem + SLOT) % (SLOTS - 3)) * SLOT;    // load
            // pure: one store a tick; every 4th tick it is a data slot, the others re-point
            // the pc, the read address and the write address, which is what the wide form
            // does inside one tick
            float* to = mem + ((wide || t % 4 == 0) ? 3 + where(mem + 2 * SLOT) % (SLOTS - 3)
                                                    : (size_t)(t % 4 - 1)) * SLOT;
            float in[32], out[64];
            std::memcpy(in, from, SLOT * 4);
            std::memcpy(in + SLOT, to, SLOT * 4);
            apply(vop, in, out, rows);                                           // execute
            write(to, out);                                                      // store
            if (wide) {
                write(mem, out + 16);              // next pc
                write(mem + SLOT, out + 32);       // next read address
                write(mem + 2 * SLOT, out + 48);   // next write address
            }
        }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double sum = 0.0;
    for (auto& r : rs) {
        for (int i = 0; i < SLOTS * SLOT; i++) sum += r.mem[i];
        std::free(r.mem);
    }
    std::free(bank);
    std::printf("  %-5s %8zu vops  %2d run%s  %7.1f ns a tick  %6.1f million ticks a second"
                "   jumped to another vop on %.0f%% of ticks (sum %.2f)\n", wide ? "wide" : "pure", count,
                runs, runs == 1 ? " " : "s", secs * 1e9 / ((double)ticks * runs),
                (double)ticks * runs / secs / 1e6, 100.0 * jumps / ((double)ticks * runs), sum);
    return secs * 1e9 / ((double)ticks * runs);
}

int main(int argc, char** argv) {
    size_t mb = argc > 1 ? (size_t)std::atol(argv[1]) : 1;
    int runs = argc > 2 ? std::atoi(argv[2]) : 1;
    long ticks = argc > 3 ? std::atol(argv[3]) : 2000000;
    std::printf("bank of %zu MB, %d run%s sharing it\n", mb, runs, runs == 1 ? "" : "s");
    double p = bench(false, mb << 20, runs, ticks);
    double w = bench(true, mb << 20, runs, ticks);
    std::printf("  a wide tick costs %.2f pure ticks and does the work of about 4\n", w / p);
    return 0;
}
