"""Control on the frozen core, trained without labels: balancing a pole, then harder.

  python -m vvm2.control WORLD [ROUNDS] [key=value ...]
  WORLD:   pole    the cart and pole, all four state numbers shown
           blind   the same with the two velocities hidden
           swing   the pendulum swing-up
  options: who=vvm,mlp,gru   which models (mlp2: an MLP with the machine's multiply-adds)
           train=es          an evolution strategy, the machine run with whole steps
           train=pg          a policy gradient through the soft machine (pole and blind)
           seed=0  layout=one|each  ticks=1 (machine ticks a world step)
           ops=4 k=2 reads=2 scratch=4      the machine
           pop=32 eps=4 sigma=0.05 alpha=1  the evolution strategy
           batch=16 lr=0.003                the policy gradient
  python -m vvm2.control --quick     pole, 3 rounds of each kind of training, tiny

The worlds are in vvm2/worlds.py (the constants are in their headers).

The loop. The driver writes the state numbers raw into the input slots (layout 'one': one
slot, the numbers in its first places; 'each': a slot a number) before every tick, and
every `ticks` ticks the world reads the first two numbers of the one output slot: the cart
is pushed right when the first is the larger (the pendulum's torque is twice their
difference). The machine is started once an episode and runs to the end of it; nothing is
wiped between world steps, so the pc slot, the scratch slots and the pointers carry over.

What the core lacks for this, and the way round it. `Core.forward` calls `world(t)` for the
input but does not hand it the output slots, and a control loop needs them to make the next
state. Nor can the world end a run early. The core is frozen, so `machine_memory` reads the
core's memory out of the running `forward` (its Python frame) and `Stop` is raised to end a
run when every episode is over. The machine itself is untouched and unhelped.

Training. Evolution strategy: each round, `pop` copies of the weights with noise added
(mirrored pairs, noise `sigma`), each scored on the same `eps` fresh episodes; the weights
move by `alpha * sigma` times the rank-weighted mean of the noise. Forward runs only, the
machine with whole steps throughout. Policy gradient: the push is sampled from the two
output numbers (times one learned sharpness), the machine runs soft, and the gradient of
log-probability times reward-to-go (less a per-step average) goes back through the whole
episode. The MLP and the GRU (one GRU step a world step) are trained by the same code.
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .common import mlp_matched, params
from .core import SLOT, Core
from .curves import MOVES, lay, place
from .worlds import CartPole, Pendulum


class Stop(Exception):
    """Raised by the world to end the machine's run: every episode is over."""


def machine_memory():
    """The running core's memory [B, slots, 16], read from the frame of Core.forward (which
    called the world function, which called this). Read only."""
    return sys._getframe(2).f_locals["mem"]


class VMPolicy(nn.Module):
    def __init__(self, n_obs, layout="one", ticks=1, **core):
        super().__init__()
        self.core = Core(**core)
        self.layout, self.ticks = layout, ticks
        self.n_in = 1 if layout == "one" else n_obs

    def rollout(self, env, pick, stop=True, watch=None):
        """One batch of episodes, the machine running throughout. pick: output numbers ->
        action. watch: called with the memory at every world step."""
        def world(t):
            if t and t % self.ticks == 0:
                mem = machine_memory()
                if watch is not None:
                    watch(mem)
                env.step(pick(mem[:, -1, :2]))
                if stop and env.over():
                    raise Stop
            return lay(env.obs(), self.layout)

        try:
            outs = self.core(world, self.ticks * env.cap, 1, trace=watch is not None)
            env.step(pick(outs[-1][:, 0, :2]))
        except Stop:
            pass

    def madds(self):
        c = self.core
        return self.ticks * (c.ops * SLOT + c.k * c.fan_out * (c.reads + 2) * SLOT)


class MLPPolicy(nn.Module):
    def __init__(self, n_obs, target):
        super().__init__()
        self.net = mlp_matched(n_obs, 2, target)

    def rollout(self, env, pick, **_):
        while not env.over():
            env.step(pick(self.net(env.obs())))

    def madds(self):
        return self.net.madds_total()


class GRUPolicy(nn.Module):
    def __init__(self, n_obs, target):
        super().__init__()
        def size(w):
            return 3 * (w * n_obs + w * w + 2 * w) + 2 * w + 2
        self.width = min(range(2, 400), key=lambda w: abs(size(w) - target))
        self.cell = nn.GRUCell(n_obs, self.width)
        self.out = nn.Linear(self.width, 2)

    def rollout(self, env, pick, **_):
        h = torch.zeros(len(env.score), self.width)
        while not env.over():
            h = self.cell(env.obs(), h)
            env.step(pick(self.out(h)))

    def madds(self):
        return params(self)


def play(policy, make_env, n, seed):
    """n fresh episodes, the action taken outright. Returns the world."""
    env = make_env(n, torch.Generator().manual_seed(seed))
    policy.eval()
    with torch.no_grad():
        policy.rollout(env, env.greedy)
    return env


def tied_ranks(f):
    """Ranks from -0.5 (worst) to 0.5 (best); equal scores share a rank."""
    less = (f[None, :] < f[:, None]).float().sum(1)
    same = (f[None, :] == f[:, None]).float().sum(1) - 1
    return (less + 0.5 * same) / (len(f) - 1) - 0.5


def es(policy, make_env, rounds, pop=32, eps=4, sigma=0.05, alpha=1.0, seed=0, goal=None,
       say=print):
    """The evolution strategy. Returns (episodes used, episodes when the unperturbed weights
    first scored `goal` on 16 fresh episodes, or None). Leaves the policy holding the
    weights that scored best on those checks (the latest of equals)."""
    g = torch.Generator().manual_seed(seed)
    theta = parameters_to_vector(policy.parameters()).detach().clone()
    used, first, best, keep, full = 0, None, None, theta, 0
    for r in range(rounds):
        noise = torch.randn(pop // 2, len(theta), generator=g)
        fit = torch.zeros(2, pop // 2)
        for i in range(pop // 2):
            for j, sign in enumerate((1.0, -1.0)):
                vector_to_parameters(theta + sign * sigma * noise[i], policy.parameters())
                fit[j, i] = play(policy, make_env, eps, 7919 * seed + r).score.mean()
        rank = tied_ranks(fit.flatten()).view(2, -1)
        pull = rank[0] - rank[1]
        theta = theta + alpha * sigma * (pull @ noise) / pull.abs().sum().clamp_min(1e-6)
        vector_to_parameters(theta, policy.parameters())
        env = play(policy, make_env, 16, 104729 * (seed + 1) + r)
        check = env.score
        used += pop * eps + 16
        if best is None or check.mean() >= best:
            best, keep = check.mean(), theta.clone()
        if goal is not None and first is None and check.mean() >= goal:
            first = used
        say("    round %3d: copies %.1f (best %.1f), the weights themselves %.1f; %d episodes"
            % (r, fit.mean(), fit.max(), check.mean(), used))
        full = full + 1 if goal is not None and check.min() >= env.cap else 0
        if full >= 3:      # every check episode ran to the cap three rounds running
            break
    vector_to_parameters(keep, policy.parameters())
    return used, first


def pg(policy, make_env, rounds, batch=16, lr=3e-3, seed=0, goal=None, say=print):
    """The policy gradient (pole worlds only: the action is a push sampled left or right).
    Returns (episodes used, episodes when it first scored `goal` acting outright)."""
    g = torch.Generator().manual_seed(seed)
    temp = nn.Parameter(torch.tensor(5.0))
    opt = torch.optim.Adam(list(policy.parameters()) + [temp], lr=lr)
    used, first = 0, None
    for r in range(rounds):
        policy.train()
        env = make_env(batch, g)
        logp, alive = [], []

        def pick(two):
            logit = temp * (two[:, 0] - two[:, 1])
            right = torch.rand(len(logit), generator=g) < torch.sigmoid(logit.detach())
            logp.append(-F.binary_cross_entropy_with_logits(logit, right.float(), reduction="none"))
            alive.append(env.alive.float())
            return right

        policy.rollout(env, pick)
        live = torch.stack(alive)                                   # [steps, batch]
        togo = live.flip(0).cumsum(0).flip(0)                       # steps still to come
        base = (togo * live).sum(1, keepdim=True) / live.sum(1, keepdim=True).clamp_min(1)
        gain = (togo - base) / togo.std().clamp_min(1.0)
        loss = -(torch.stack(logp) * gain * live).sum() / batch
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        used += batch
        if r % 10 == 9 or r == rounds - 1:
            check = play(policy, make_env, 16, 104729 * (seed + 1) + r).score.mean()
            if goal is not None and first is None and check >= goal:
                first = used
            say("    round %3d: while exploring %.1f; acting outright %.1f; %d episodes"
                % (r, env.score.mean(), check, used))
    return used, first


def look(policy, make_env, seed, show=24):
    """What the trained machine does tick by tick, and whether it is a reflex."""
    core, n_in = policy.core, policy.n_in
    env = make_env(16, torch.Generator().manual_seed(seed))
    seen, acts = [], []

    def watch(mem):
        seen.append((env.obs().clone(), mem[:, 0].clone(), env.alive.clone(),
                     mem[:, 1 + n_in:-1].abs().amax(dim=(1, 2))))

    def pick(two):
        acts.append(env.greedy(two))
        return acts[-1]

    policy.eval()
    with torch.no_grad():
        policy.rollout(env, pick, stop=False, watch=watch)
    acts = acts[:len(seen)]                 # the last push comes after the machine's run
    live = torch.stack([s[2] for s in seen])                         # [steps, 16]
    tick_live = live.repeat_interleave(policy.ticks, dim=0)          # [ticks, 16]
    log = core.trace[:len(tick_live)]
    print("  trace: episode 0 of 16 (it balanced %d steps), the first %d ticks, %d tick%s a "
          "world step. After each tick: the vop that ran, where the pointers are, their move"
          % (env.score[0], show, policy.ticks, "" if policy.ticks == 1 else "s"))
    print("    vop:    " + " ".join("%5d" % int(t[0][0]) for t in log[:show]))
    for p in range(core.reads + 1):
        name = "write:  " if p == core.reads else "read %d: " % (p + 1)
        print("    " + name + " ".join(
            "%5s" % place(int(t[1][0, p].argmax()), n_in, core.scratch) for t in log[:show]))
        print("      did:  " + " ".join("%5s" % MOVES[int(t[2][0, p].argmax())] for t in log[:show]))
    if acts[0].dtype == torch.bool:
        print("    push:   " + " ".join("%5s" % ("R" if a[0] else "L") for a in acts[:show]))
    else:
        print("    torque: " + " ".join("%5.1f" % a[0] for a in acts[:show]))
    at = torch.stack([t[1].argmax(dim=-1) for t in log])             # [ticks, 16, P]
    edges = torch.tensor([1, 1 + n_in, 1 + n_in + core.scratch])
    region = torch.bucketize(at, edges, right=True)                  # 0 pc 1 in 2 scratch 3 out
    for p in range(core.reads + 1):
        share = [(region[..., p][tick_live] == k).float().mean() for k in range(4)]
        print("    %s share of ticks on the pc slot %.2f, input %.2f, scratch %.2f, output %.2f"
              % ("write pointer:" if p == core.reads else "read pointer %d:" % (p + 1), *share))
    picks = torch.stack([t[0] for t in log])[tick_live]
    count = torch.bincount(picks, minlength=core.ops).float()
    prob = count / count.sum()
    top = torch.stack([t[0] for t in log])
    change = ((top[1:] != top[:-1]) & tick_live[1:]).float().sum() / tick_live[1:].sum()
    print("    bank: %.2f of %d vops used, entropy %.2f, busiest %.2f; the pick differs from "
          "the tick before on %.2f of ticks" % (
              (count > 0).float().mean(), core.ops,
              -(prob[prob > 0] * prob[prob > 0].log()).sum() / torch.tensor(float(max(core.ops, 2))).log(),
              prob.max(), change))
    pcs = torch.stack([s[1] for s in seen])                          # [steps, 16, 16]
    drift = (pcs[1:] - pcs[:-1]).norm(dim=-1)[live[1:]].mean()
    print("    scratch: the largest number ever in it %.3f (0: never written). The pc slot: "
          "length %.2f, moves %.3f a world step" % (
              torch.stack([s[3] for s in seen])[live].max(), pcs.norm(dim=-1)[live].mean(), drift))
    obs = torch.stack([s[0] for s in seen])[live]
    took = torch.stack(acts)[live]
    with torch.no_grad():
        fresh = env.greedy(core(lay(obs, policy.layout), 8 + policy.ticks, 1)[-1][:, 0, :2])
    if took.dtype == torch.bool:
        print("    reflex test: a machine started fresh and shown only the present state for "
              "%d ticks pushes the same way on %.3f of %d steps (right on %.2f of them)"
              % (8 + policy.ticks, (fresh == took).float().mean(), len(took), took.float().mean()))
    else:
        print("    reflex test: a machine started fresh and shown only the present state for "
              "%d ticks gives a torque %.3f away on average (range 4), over %d steps"
              % (8 + policy.ticks, (fresh - took).abs().mean(), len(took)))
    sys.stdout.flush()


def run(world, rounds, who="vvm,mlp,gru", train="es", seed=0, layout="one", ticks=1, ops=4, k=2,
        reads=2, scratch=4, pop=32, eps=4, sigma=0.05, alpha=1.0, batch=16, lr=3e-3, trace=True):
    torch.set_num_threads(4)
    kind = Pendulum if world == "swing" else CartPole
    def make_env(n, g):
        return kind(n, g, hide=world == "blind")
    n_obs = make_env(1, None).n_obs
    goal = 475.0 if kind is CartPole else None
    rows = []
    for name in who.split(","):
        torch.manual_seed(seed)
        vm = VMPolicy(n_obs, layout, ticks, ops=ops, k=min(k, ops), reads=reads, scratch=scratch)
        policy = (vm if name == "vvm" else GRUPolicy(n_obs, params(vm)) if name == "gru"
                  else MLPPolicy(n_obs, params(vm) if name == "mlp" else vm.madds()))
        if name == "vvm":
            vm.core.hard = train == "es"
        print("  %s on %s, seed %d, %s: %d weights, %d multiply-adds a world step" % (
            name, world, seed, train, params(policy), policy.madds()), flush=True)
        say = lambda s: print(s, flush=True)
        if train == "es":
            used, first = es(policy, make_env, rounds, pop, eps, sigma, alpha, seed, goal, say)
        else:
            used, first = pg(policy, make_env, rounds, batch, lr, seed, goal, say)
        tests = [("", play(policy, make_env, 100, 999 + seed))]
        if name == "vvm" and train == "pg":
            vm.core.hard = True
            tests = [(", as trained", tests[0][1]),
                     (", run with whole steps", play(policy, make_env, 100, 999 + seed))]
        for tag, env in tests:
            extra = ("%.2f" % (env.score >= env.cap).float().mean() if kind is CartPole
                     else "%.1f of the last 50 steps upright" % env.up.mean())
            rows.append("| %s%s | %s | %d | %d | %d | %.1f | %s | %s | %d |" % (
                name, tag, train, seed, params(policy), policy.madds(), env.score.mean(), extra,
                "never" if first is None else "%d" % first, used))
        if name == "vvm" and trace:
            look(policy, make_env, 5000 + seed)
    print("%s, 100 fresh test episodes. The machine: layout '%s', %d tick%s a world step, bank "
          "of %d (k %d), %d read pointers, %d scratch slots" % (
              world, layout, ticks, "" if ticks == 1 else "s", ops, min(k, ops), reads, scratch))
    print("| model | training | seed | weights | multiply-adds a world step | %s | %s | episodes "
          "to first reach 475 | episodes used |" % (
              ("steps balanced", "share reaching 500") if kind is CartPole else ("score", "end")))
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows), flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--quick"]
    extra = dict(a.split("=") for a in args[2:])
    opts = dict(who=extra.get("who", "vvm,mlp,gru"), seed=int(extra.get("seed", 0)),
                layout=extra.get("layout", "one"), ticks=int(extra.get("ticks", 1)),
                ops=int(extra.get("ops", 4)), k=int(extra.get("k", 2)),
                reads=int(extra.get("reads", 2)), scratch=int(extra.get("scratch", 4)),
                pop=int(extra.get("pop", 32)), eps=int(extra.get("eps", 4)),
                sigma=float(extra.get("sigma", 0.05)), alpha=float(extra.get("alpha", 1)),
                batch=int(extra.get("batch", 16)), lr=float(extra.get("lr", 3e-3)),
                trace=extra.get("trace", "1") == "1")
    if "--quick" in sys.argv:
        opts.update(pop=4, eps=2, batch=4)
        run("pole", 3, train="es", **opts)
        run("pole", 3, train="pg", **opts)
        sys.exit()
    run(args[0] if args else "pole", int(args[1]) if len(args) > 1 else 60,
        train=extra.get("train", "es"), **opts)
