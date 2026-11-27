// Copy a list, on the machine in vvm.hpp: a program written by hand, the speed of a tick,
// and a first try at finding the program by evolution (no gradient, forward runs only).
//
//   cpp/copy hand            the hand-written program, scored at several lengths
//   cpp/copy speed           ticks a second
//   cpp/copy evolve [GENS] [POP] [SEED] [TASK] [START] [CORES]
//       TASK   copy (lists of 1 to 8) | one | two | four (lists of exactly that many) |
//              negate (one item, its negative wanted) | reverse (lists of 1 to 8, backwards)
//              long (lists of 1 to 24: more items than there is room for vops)
//       START  random (every slot random) | pass (every program slot starts as an op that
//              passes a slot through nearly unchanged; the cores' addresses are random)
//
// Memory, 128 slots:   0..3  four cores      4..31  program and scratch (what evolution owns,
//                      32..79  input (the world's)     80..127  output      with the cores)
// A list of L items sits in the first L input slots, one item a slot; an item is one of 8
// fixed random grids (the world's code book). The answer for item i is whichever code book
// grid output slot i is nearest. The machine runs for 2 L + 4 ticks. One core runs unless said.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "vvm.hpp"

static const int N = 128, CORES = 4, OWN = 32, IN0 = 32, OUT0 = 80, CAP = 48, SYMBOLS = 8;
static float BOOK[SYMBOLS][SLOT];
static int LMIN = 1, LMAX = 8, SIGN = 1, RUNNING = 1;   // RUNNING: how many of the 4 cores tick
static bool BACKWARDS = false;

typedef std::vector<float> Genome;   // slots 0..31: the cores, the program, the scratch

static void set_up(Machine& m, const Genome& g, const int* list, int len) {
    m.n = N;
    m.mem.assign((size_t)N * SLOT, 0.0f);
    std::memcpy(m.mem.data(), g.data(), (size_t)OWN * SLOT * sizeof(float));
    m.cores.assign({0, 1, 2, 3});
    m.cores.resize((size_t)RUNNING);
    m.world_lo = IN0, m.world_hi = IN0 + CAP;
    for (int i = 0; i < len; i++) std::memcpy(m.slot(IN0 + i), BOOK[list[i]], SLOT * sizeof(float));
}

// run one list; returns how far the output slots are from what was wanted (lower is better)
// and counts the items answered right. "How far" is 1 minus the likeness of direction
// between an output slot and the wanted grid: 0 for the wanted grid, 1 for an empty slot or
// an unrelated one. (Squared error was tried first and misleads the search: writing anything
// imperfect to an output slot scores worse than leaving it empty, so it learns to write
// nothing.)
static float run(Machine& m, const Genome& g, const int* list, int len, int* right) {
    set_up(m, g, list, len);
    for (int t = 0; t < 2 * len + 4; t++) m.tick();
    float err = 0.0f;
    for (int i = 0; i < len; i++) {
        const float* out = m.slot(OUT0 + i);
        int best = 0;
        float best_d = 1e30f;
        for (int s = 0; s < SYMBOLS; s++) {
            float d = 0.0f;
            for (int q = 0; q < SLOT; q++) d += (out[q] - SIGN * BOOK[s][q]) * (out[q] - SIGN * BOOK[s][q]);
            if (d < best_d) best_d = d, best = s;
        }
        {
            const float* want = BOOK[list[BACKWARDS ? len - 1 - i : i]];
            float dot = 0.0f, a = 0.0f, b = 0.0f;
            for (int q = 0; q < SLOT; q++) dot += out[q] * SIGN * want[q], a += out[q] * out[q], b += want[q] * want[q];
            err += 1.0f - (a > 1e-12f ? dot / std::sqrt(a * b) : 0.0f);
        }
        if (right) *right += best == list[BACKWARDS ? len - 1 - i : i];
    }
    return err / (float)len;
}

static Genome by_hand() {
    Genome g((size_t)OWN * SLOT, 0.0f);
    auto at = [&](int slot, int row, int col) -> float& { return g[(size_t)slot * SLOT + row * SIDE + col]; };
    const float step = address_of(1, N);
    at(0, 0, 0) = address_of(4, N);                     // the one core starts on the vop at slot 4
    // the copy vop: slot 4 passes a slot through; slot 5 has its addresses (load the first
    // input slot, store to the first output slot, then go to the vop at slot 6) and, in its
    // second row, the amount to move by
    for (int i = 0; i < SIDE; i++) at(4, i, i) = 1.0f;
    at(5, 0, 0) = address_of(IN0, N), at(5, 0, 1) = address_of(OUT0, N), at(5, 0, 2) = address_of(6, N);
    at(5, 1, 0) = step, at(5, 1, 1) = step;
    // the bump vop: slot 6 adds the second row of what it loads to the first; slot 7 has its
    // addresses (load slot 5, store back to slot 5, then go to the vop at slot 4)
    for (int i = 0; i < SIDE; i++) at(6, i, i) = 1.0f;
    at(6, 0, 1) = 1.0f;
    at(7, 0, 0) = address_of(5, N), at(7, 0, 1) = address_of(5, N), at(7, 0, 2) = address_of(4, N);
    return g;
}

static void report(const Genome& g, const char* name, unsigned seed) {
    Machine m;
    std::mt19937 rng(seed);
    std::printf("%s: share of items right / share of whole lists right\n", name);
    for (int len : {LMAX, 2 * LMAX, 48}) {
        int right = 0, whole = 0, lists = 500;
        for (int k = 0; k < lists; k++) {
            int list[CAP], r = 0;
            for (int i = 0; i < len; i++) list[i] = (int)(rng() % SYMBOLS);
            run(m, g, list, len, &r);
            right += r, whole += r == len;
        }
        std::printf("  length %2d: %.3f / %.3f\n", len, right / (double)(lists * len), whole / (double)lists);
    }
}

int main(int argc, char** argv) {
    std::mt19937 book_rng(7);
    std::uniform_real_distribution<float> u(-0.9f, 0.9f);
    for (auto& s : BOOK) for (auto& x : s) x = u(book_rng);
    const std::string what = argc > 1 ? argv[1] : "hand";

    if (what == "hand") { report(by_hand(), "the program written by hand", 1); return 0; }

    if (what == "speed") {
        RUNNING = 4;
        Machine m;
        int list[CAP];
        for (int i = 0; i < CAP; i++) list[i] = i % SYMBOLS;
        set_up(m, by_hand(), list, CAP);
        const long ticks = 20000000;
        auto t0 = std::chrono::steady_clock::now();
        for (long t = 0; t < ticks; t++) {
            if (t % 40 == 0) set_up(m, by_hand(), list, CAP);
            m.tick();
        }
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("one thread: %.1f million ticks a second (4 cores a tick, so %.1f million ops a second)\n",
                    ticks / s / 1e6, 4 * ticks / s / 1e6);
        return 0;
    }

    // evolution across a population: each individual is what slots 0..31 hold at the start
    const int gens = argc > 2 ? std::atoi(argv[2]) : 2000, pop = argc > 3 ? std::atoi(argv[3]) : 512;
    const unsigned seed = argc > 4 ? (unsigned)std::atoi(argv[4]) : 0;
    const std::string task = argc > 5 ? argv[5] : "copy", start = argc > 6 ? argv[6] : "random";
    if (argc > 7) RUNNING = std::atoi(argv[7]);
    if (task == "one") LMIN = LMAX = 1;
    if (task == "two") LMIN = LMAX = 2;
    if (task == "four") LMIN = LMAX = 4;
    if (task == "negate") LMIN = LMAX = 1, SIGN = -1;
    if (task == "reverse") BACKWARDS = true;
    if (task == "long") LMAX = 24;        // lists of 1 to 24: more items than vops fit
    const int keep = pop / 4, lists = 16;
    std::mt19937 rng(seed);
    std::vector<Genome> people((size_t)pop, Genome((size_t)OWN * SLOT));
    for (auto& g : people) {
        std::uniform_real_distribution<float> w(-0.5f, 0.5f), a(0.0f, 1.0f);
        for (auto& x : g) x = w(rng);
        if (start == "pass")      // even slots: ops that pass a slot through; odd: addresses
            for (int sl = CORES; sl < OWN; sl += 2)
                for (int q = 0; q < SLOT; q++) g[(size_t)sl * SLOT + q] = (q / SIDE == q % SIDE ? 1.0f : 0.0f) + 0.1f * w(rng);
        for (int sl = 0; sl < OWN; sl++)
            if (sl < CORES || sl % 2) for (int k = 0; k < 3; k++) g[(size_t)sl * SLOT + k] = a(rng);
    }
    std::vector<float> score((size_t)pop);
    std::vector<int> order((size_t)pop);
    std::printf("evolving %s (start: %s): %d individuals, %d generations, seed %u; lists of %d to %d "
                "items; %d cores running\n", task.c_str(), start.c_str(), pop, gens, seed, LMIN, LMAX, RUNNING);
    for (int gen = 0; gen <= gens; gen++) {
        int batch[16][CAP], lens[16];
        for (int k = 0; k < lists; k++) {
            lens[k] = LMIN + (int)(rng() % (LMAX - LMIN + 1));
            for (int i = 0; i < lens[k]; i++) batch[k][i] = (int)(rng() % SYMBOLS);
        }
#pragma omp parallel
        {
            Machine m;
#pragma omp for schedule(static)
            for (int p = 0; p < pop; p++) {
                float err = 0.0f;
                for (int k = 0; k < lists; k++) err += run(m, people[p], batch[k], lens[k], nullptr);
                score[p] = err / lists;
            }
        }
        for (int p = 0; p < pop; p++) order[p] = p;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return score[a] < score[b]; });
        if (gen % std::max(1, gens / 20) == 0 || gen == gens) {
            Machine m;
            int right = 0, total = 0;
            for (int k = 0; k < lists; k++) run(m, people[order[0]], batch[k], lens[k], &right), total += lens[k];
            std::printf("  generation %5d: best error %.4f, items right %.2f\n", gen, score[order[0]],
                        right / (double)total);
            std::fflush(stdout);
        }
        if (gen == gens) break;
        std::vector<Genome> next((size_t)pop);
        for (int p = 0; p < keep; p++) next[p] = people[order[p]];
        std::normal_distribution<float> bell(0.0f, 1.0f);
        static const float sizes[] = {0.3f, 0.1f, 0.03f, 0.01f, 0.003f, 0.001f};
        for (int p = keep; p < pop; p++) {
            Genome child = next[rng() % keep];
            if (rng() % 4 == 0) {                       // take some whole slots from another
                const Genome& other = next[rng() % keep];
                for (int s = 0; s < OWN; s++)
                    if (rng() % 2) std::copy(other.begin() + s * SLOT, other.begin() + (s + 1) * SLOT, child.begin() + s * SLOT);
            }
            const int changes = 1 + (int)(rng() % 3);
            for (int c = 0; c < changes; c++) {
                const int s = (int)(rng() % OWN), kind = (int)(rng() % 8);
                float* slot = child.data() + (size_t)s * SLOT;
                const float size = sizes[rng() % 6];
                if (kind == 0) {                        // copy another slot over this one
                    const int from = (int)(rng() % OWN);
                    std::copy(child.begin() + from * SLOT, child.begin() + (from + 1) * SLOT, slot);
                } else if (kind <= 3) {                 // change one number
                    float& x = slot[rng() % (rng() % 2 ? 3 : SLOT)];
                    x = std::min(1.0f, std::max(-1.0f, x + size * bell(rng)));
                } else {                                // change every number a little
                    for (int q = 0; q < SLOT; q++) slot[q] = std::min(1.0f, std::max(-1.0f, slot[q] + size * bell(rng)));
                }
            }
            next[p] = std::move(child);
        }
        people.swap(next);
    }
    report(people[order[0]], "the best individual found", 1);
    return 0;
}
