"""The lights game programmed by hand: two vops, weights written here, nothing trained.

    python -m old.simon_hand7
    python -m old.simon_hand7 noise=0.3 steps=1500    does training keep it? (see settle)

This answers one question: does the routine fit in the machine as it is? The program is a
tape recorder.

    vop W (runs while the signal says watch; the pc starts on it)
        store    what read pointer 1 is on (the light)
        write pointer: next          read 1: stay on the light
        when read pointer 2 (on the signal) shows "play" instead:
            read 1 -> first scratch slot,  write pointer -> the button,  pc -> vop P
    vop P (runs from then on)
        store    what read pointer 1 is on (a scratch slot), to the button
        read 1: next                 write pointer: stay

A light held for several ticks is stored that many times and played back that many times,
so the same two vops play any hold; it costs scratch slots, not counting.
"""
import torch
import torch.nn.functional as F

from .core7 import SLOT
from .simon7 import SimonVM, score

STAY, NEXT, SC0, OUT0 = 0, 1, 3 + 2, 3 + 4   # a pointer's choices: stay, the two steps,
LOUD = 20.0                                   # then in0 inN sc0 scN out0 outN, then seek


def program(vm):
    core, book = vm.core, vm.book
    watch_play = torch.linalg.pinv(book[-2:])        # [16, 2]: reads "watch" and "play" apart
    is_watch, is_play = watch_play[:, 0], watch_play[:, 1]
    g = torch.Generator().manual_seed(3)
    pcs = torch.randn(2, SLOT, generator=g)
    pcs = F.normalize(pcs - pcs.mean(dim=1, keepdim=True), dim=1)   # the two pc values
    one = torch.eye(SLOT)
    r1, r2 = slice(0, SLOT), slice(SLOT, 2 * SLOT)   # what read 1 and read 2 loaded
    value, next_pc, steps = slice(0, SLOT), slice(SLOT, 2 * SLOT), 2 * SLOT

    def want(p, move):   # the row of the output that asks pointer p for a move
        return steps + p * core.steps + move

    with torch.no_grad():
        for w in (core.m, core.bias, core.keys, core.boot_at):
            w.zero_()
        core.boot_pc.copy_(pcs[0] * 0.5)
        core.keys[:2] = F.layer_norm(pcs, (SLOT,)) * 1.25
        core.boot_at[0, 0] = core.boot_at[1, 1] = core.boot_at[2, 2] = LOUD   # in0, inN, sc0
        W, P = 0, 1
        core.m[W, value, r1] = one
        core.m[W, next_pc, r2] = torch.outer(pcs[1], is_play)
        core.m[W, want(0, STAY), r2] = LOUD * is_watch
        core.m[W, want(0, SC0), r2] = LOUD * is_play
        core.bias[W, want(1, STAY)] = LOUD
        core.m[W, want(2, NEXT), r2] = LOUD * is_watch
        core.m[W, want(2, OUT0), r2] = LOUD * is_play
        core.m[P, value, r1] = one
        core.bias[P, want(0, NEXT)] = LOUD
        core.bias[P, want(1, STAY)] = LOUD
        core.bias[P, want(2, STAY)] = LOUD


def settle(noise, steps, most=6, seed=0):
    """Does training keep the routine? The two vops go into a bank of 128; every weight then
    gets random noise, `noise` times the size an untrained machine starts with; then it is
    trained as usual on games of 1 to `most` lights."""
    torch.manual_seed(seed)
    vm = SimonVM(4, 1, 1, scratch=64, ops=128, k=4)
    program(vm)
    with torch.no_grad():
        for w, size in ((vm.core.m, 0.125), (vm.core.bias, 0.1), (vm.core.keys, 0.25)):
            w.add_(torch.randn_like(w) * size * noise)
    tests, g = (6, 12, 24, 48), torch.Generator().manual_seed(seed)

    def both(when):
        vm.eval()
        soft = score(vm, tests, torch.Generator().manual_seed(9), 4)
        vm.core.hard = True
        hard = score(vm, tests, torch.Generator().manual_seed(9), 4)
        vm.core.hard = False
        print("| %s, as trained | %s |\n| %s, whole steps | %s |" % (when, soft, when, hard),
              flush=True)

    print("the hand routine in a bank of 128 with noise %.2f, then %d training steps on games "
          "of 1 to %d lights" % (noise, steps, most))
    print("| when | " + " | ".join("length %d" % t for t in tests) + " |\n|---|---|---|---|---|")
    both("noise added")
    opt = torch.optim.AdamW(vm.parameters(), lr=5e-4, weight_decay=0.01)
    for step in range(steps):
        vm.train()
        x = torch.randint(4, (64, int(torch.randint(1, most + 1, (1,), generator=g))), generator=g)
        loss = F.cross_entropy(vm(x).reshape(-1, 4), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vm.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 4) == 0:
            print("    step %d: loss %.3f" % (step, loss.item()), flush=True)
    both("after training")


if __name__ == "__main__":
    import sys
    torch.set_num_threads(1)
    a = dict(s.split("=") for s in sys.argv[1:])
    if "noise" in a:
        settle(float(a["noise"]), int(a.get("steps", 1500)))
        sys.exit()
    print("the lights game by hand: two vops, nothing trained. Share of lights right / share "
          "of whole games right, run with whole steps")
    print("| ticks a light | length 6 | length 12 | length 24 | length 48 |")
    print("|---|---|---|---|---|")
    for hold in (1, 2, 4, 8):
        vm = SimonVM(4, hold, 1, scratch=48 * hold + 2, ops=4, k=2)
        program(vm)
        vm.eval()
        vm.core.hard = True
        print("| %d | %s |" % (hold, score(vm, (6, 12, 24, 48), torch.Generator().manual_seed(0), 4)),
              flush=True)
