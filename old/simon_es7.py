"""The lights game trained by evolution: the exact machine, forward only, no gradient.

    python -m old.simon_es7 [ROUNDS] [ops=4 max=6 pop=32 sigma=0.05 seed=0]

Each round: 2 * pop copies of the weights, each with random noise added (in pairs, +noise
and -noise), all run with whole steps on the same batch of games; the weights then move
toward the noise that scored better (by rank). The score is how surely the right lights are
answered. The machine that is trained is exactly the machine that runs.
"""
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from vvm2.common import params
from .simon7 import SimonVM, score, show_trace


def run(rounds=1500, ops=4, most=6, pop=32, sigma=0.05, seed=0, colours=4):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    vm = SimonVM(colours, 1, 1, scratch=64, ops=ops, k=min(4, ops))
    vm.eval()
    vm.core.hard = True
    w = parameters_to_vector(vm.parameters()).detach().clone()
    mean, var, g = torch.zeros_like(w), torch.zeros_like(w), torch.Generator().manual_seed(seed)
    tests = (most, 2 * most, 4 * most)
    print("simon by evolution: bank of %d (%d weights), games of 1 to %d lights, %d copies a "
          "round, noise %.3f. Share of lights right / share of whole games right, whole steps"
          % (ops, params(vm), most, 2 * pop, sigma), flush=True)

    def fitness(vec, x):
        vector_to_parameters(vec, vm.parameters())
        return -F.cross_entropy(vm(x).reshape(-1, colours), x.reshape(-1)).item()

    with torch.no_grad():
        for r in range(rounds):
            x = torch.randint(colours, (64, int(torch.randint(1, most + 1, (1,), generator=g))), generator=g)
            noise = torch.randn(pop, len(w), generator=g)
            f = torch.tensor([[fitness(w + sigma * e, x), fitness(w - sigma * e, x)] for e in noise])
            rank = f.flatten().argsort().argsort().float().view(pop, 2) / (2 * pop - 1) - 0.5
            grad = -((rank[:, 0] - rank[:, 1]).unsqueeze(1) * noise).mean(dim=0) / sigma
            mean = 0.9 * mean + 0.1 * grad                      # Adam on the estimated slope
            var = 0.999 * var + 0.001 * grad * grad
            w -= 0.01 * (mean / (1 - 0.9 ** (r + 1))) / ((var / (1 - 0.999 ** (r + 1))).sqrt() + 1e-8)
            if r % max(1, rounds // 15) == 0 or r == rounds - 1:
                vector_to_parameters(w, vm.parameters())
                print("    round %d: lengths %s: %s" % (r, ", ".join(map(str, tests)), score(
                    vm, tests, torch.Generator().manual_seed(5), colours)), flush=True)
    vector_to_parameters(w, vm.parameters())
    show_trace(vm, torch.randint(colours, (8, min(4, most)), generator=torch.Generator().manual_seed(1)), hard=True)


if __name__ == "__main__":
    a = dict(s.split("=") for s in sys.argv[1:] if "=" in s)
    first = [s for s in sys.argv[1:] if "=" not in s]
    run(rounds=int(first[0]) if first else 1500, ops=int(a.get("ops", 4)), most=int(a.get("max", 6)),
        pop=int(a.get("pop", 32)), sigma=float(a.get("sigma", 0.05)), seed=int(a.get("seed", 0)))
