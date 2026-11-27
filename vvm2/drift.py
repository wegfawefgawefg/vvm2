"""Does the copy loop lose its place as it goes? A check of one claim.

  python -m vvm2.drift [STEPS]

Trains the copy machine with lists starting anywhere (as scripts/tape.sh copy 3000 shift=16)
and then runs it on lists of 8, 12, 16 and 24. For each stretch of four places it reports:

  right        the share of items copied right
  one late / one early   the share of outputs that are the item one place before or after
               the right one (counted only where that item differs from the right one; a
               machine guessing would score 1 in 8)
  read - write   while the write pointer is on a place in this stretch, how far ahead of it
               the centre of the read pointer is, in places (a locked pair stays constant)
  read spread, write spread   how many slots each pointer is smeared over there

If the loop loses its place bit by bit, "right" falls gradually along the list, "one late"
or "one early" rises as it does, and the gap between the pointers or their spread grows.
If it simply stops at some place, "right" falls off a cliff and the neighbours stay at 1 in 8.
"""
import sys

import torch
import torch.nn.functional as F

from .tape import SYMBOLS, TapeVM


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    torch.set_num_threads(8)
    torch.manual_seed(0)
    vm = TapeVM()
    opt = torch.optim.AdamW(vm.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        vm.train()
        length = int(torch.randint(2, 9, (1,), generator=g))
        x = torch.randint(SYMBOLS, (64, length), generator=g)
        start = int(torch.randint(17, (1,), generator=g))
        loss = F.cross_entropy(vm.lists(x, start).reshape(-1, SYMBOLS), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vm.parameters(), 1.0)
        opt.step()
        sched.step()
    vm.eval()
    core = vm.core
    for length in (8, 12, 16, 24):
        x = torch.randint(SYMBOLS, (500, length), generator=g)
        with torch.no_grad():
            out = vm.lists(x, trace=True).argmax(dim=-1)
        a, b = core.head, core.head + length + core.scratch
        place = torch.arange(length).float()
        gap = [[] for _ in range(length)]
        rs = [[] for _ in range(length)]
        ws = [[] for _ in range(length)]
        for _, reads, write in core.trace:
            r, w = reads[:, 0, a:a + length], write[:, b:b + length]
            ok = (r.sum(dim=1) > 0.5) & (w.sum(dim=1) > 0.5)
            rn, wn = r / r.sum(dim=1, keepdim=True), w / w.sum(dim=1, keepdim=True)
            at = w.argmax(dim=1)
            spread = lambda p: torch.exp(-(p * torch.log(p.clamp_min(1e-9))).sum(dim=1))
            d, sr, sw = rn @ place - wn @ place, spread(rn), spread(wn)
            for p in range(length):
                m = ok & (at == p)
                if m.any():
                    gap[p].append(d[m].mean()); rs[p].append(sr[m].mean()); ws[p].append(sw[m].mean())
        print("copy, a list of %d (trained on 2 to 8): whole lists right %.2f" % (
            length, (out == x).all(dim=1).float().mean()))
        print("| places | right | one late | one early | read - write | read spread | write spread |")
        print("|---|---|---|---|---|---|---|")
        for lo in range(0, length, 4):
            hi = min(lo + 4, length)
            right = (out[:, lo:hi] == x[:, lo:hi]).float().mean()

            def neighbour(shift):
                hit, n = 0.0, 0
                for p in range(lo, hi):
                    q = p + shift
                    if 0 <= q < length:
                        differs = x[:, q] != x[:, p]
                        hit += (out[:, p] == x[:, q])[differs].float().sum().item()
                        n += int(differs.sum())
                return hit / max(n, 1)

            mean = lambda rows: (sum(sum(rows[p]) / len(rows[p]) for p in range(lo, hi) if rows[p])
                                 / max(1, sum(1 for p in range(lo, hi) if rows[p])))
            print("| %d to %d | %.2f | %.2f | %.2f | %+.2f | %.1f | %.1f |" % (
                lo, hi - 1, right, neighbour(-1), neighbour(1), float(mean(gap)),
                float(mean(rs)), float(mean(ws))), flush=True)
            if right < 0.9:   # where it goes wrong: whose item did it write instead?
                far = sorted(((neighbour(d), d) for d in range(-length + 1, length) if d != 0),
                             reverse=True)[:3]
                print("|   wrong items here are most often the item %s |" % ", ".join(
                    "%+d places away (%.2f)" % (d, v) for v, d in far), flush=True)


if __name__ == "__main__":
    main()
