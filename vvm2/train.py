"""One training run, one results row.

  python -m vvm2.train --task assoc --model machine --seed 0 --out out/x.jsonl

A baseline is sized either by --width or by --match params|madds, which builds the machine
described by the machine flags on the same task and sizes the baseline to its parameter
count or to its multiply-adds per example. The row (one JSON line appended to --out) holds
the settings, both counts, the accuracies and the bank-use numbers. A run whose settings
are already in the file is skipped unless --force is given.
"""
import argparse
import hashlib
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .bank import use_stats
from .baselines import GRU, MLP, Transformer, count_madds, count_params, fit_width
from .machine import Machine
from .probe import look_inside, walk_loss
from .tasks import make_task


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    a = p.add_argument
    # task
    a("--task", required=True)
    for name, kind in [("items", int), ("digits", int), ("base", int), ("classes", int),
                       ("p", int), ("bits", int), ("train_share", float), ("hops", int),
                       ("holdout", float), ("pairs", int)]:
        a("--" + name, type=kind, default=None)
    # model
    a("--model", default="machine", choices=["machine", "mlp", "gru", "transformer"])
    a("--width", type=int, default=64, help="baseline width when --match is not given")
    a("--match", default="", choices=["", "params", "madds"])
    a("--depth", type=int, default=2, help="mlp layers before the readout")
    a("--layers", type=int, default=2, help="transformer blocks")
    a("--ponder", type=int, default=0, help="gru: extra steps with a zero input")
    # machine
    a("--d", type=int, default=64)
    a("--regs", type=int, default=1)
    a("--ticks", type=int, default=2)
    a("--feed", default="whole", choices=["whole", "seq"])
    a("--hold", type=int, default=0, help="1: write the input on every tick, not the first")
    a("--alu", type=int, default=128, help="ALU hidden width")
    a("--entries", type=int, default=4096)
    a("--heads", type=int, default=2)
    a("--k", type=int, default=8)
    a("--d_key", type=int, default=32)
    a("--qnorm", default="layer", choices=["layer", "l2", "batch", "none"])
    a("--value_std", type=float, default=0.1)
    a("--no_bank", type=int, default=0)
    a("--no_alu", type=int, default=0)
    a("--mem", type=int, default=0, help="1: with the data memory")
    a("--dense_bank", type=int, default=0, help="1: dense value gradients (for checks)")
    # noise switches (training only)
    a("--pick_from", type=int, default=0, help=">k: exact top pick_from, keep k at random")
    a("--gumbel", type=float, default=0.0, help="Gumbel noise scale on sub-key scores")
    a("--qnoise", type=float, default=0.0, help="Gaussian noise std on the query")
    a("--snoise", type=float, default=0.0, help="Gaussian noise on the state, in its rms")
    # training
    a("--steps", type=int, default=2000)
    a("--batch", type=int, default=256)
    a("--lr", type=float, default=3e-3)
    a("--lr_bank", type=float, default=3e-2)
    a("--wd", type=float, default=0.0)
    a("--vdecay", type=float, default=0.0,
      help="decay of the bank values that were read, times the dense lr (Bank.decay_read)")
    a("--warmup", type=int, default=100)
    a("--seed", type=int, default=0)
    a("--threads", type=int, default=4)
    a("--evals", type=int, default=4, help="points on the accuracy curve")
    a("--tick_loss", type=int, default=0,
      help="1: chase, machine only: supervise the readout after every tick (probe.walk_loss)")
    # output
    a("--out", default="out/runs.jsonl")
    a("--label", default="")
    a("--force", type=int, default=0)
    return p.parse_args(argv)


def machine_from(args, task):
    return Machine(
        task["vocab"], task["classes"], d=args.d, regs=args.regs, ticks=args.ticks,
        feed=args.feed, hold=bool(args.hold), alu=args.alu, use_bank=not args.no_bank,
        use_alu=not args.no_alu, use_mem=bool(args.mem), snoise=args.snoise,
        entries=args.entries, heads=args.heads, k=args.k, d_key=args.d_key,
        qnorm=args.qnorm, pick_from=args.pick_from, gumbel=args.gumbel,
        qnoise=args.qnoise, value_std=args.value_std, sparse=not args.dense_bank)


def baseline_from(args, task, width):
    v, n, c = task["vocab"], task["length"], task["classes"]
    if args.model == "mlp":
        return MLP(v, n, c, width, depth=args.depth,
                   shared_vocab=task.get("shared_vocab", False))
    if args.model == "gru":
        return GRU(v, n, c, width, ponder=args.ponder)
    return Transformer(v, n, c, width, layers=args.layers)


def build(args, task):
    """The model, and the reference machine's counts if the model was matched to one."""
    if args.model == "machine":
        return machine_from(args, task), None
    ref = None
    width = args.width
    if args.match:
        machine = machine_from(args, task)
        ref = {"params": count_params(machine), "madds": count_madds(machine, task["length"])}
        measure = count_params if args.match == "params" else (
            lambda m: count_madds(m, task["length"]))
        step = 4 if args.model == "transformer" else 2
        width = fit_width(lambda w: baseline_from(args, task, w), measure,
                          ref[args.match], step=step)
    return baseline_from(args, task, width), ref


def optimisers(model, args):
    """AdamW for the dense parameters (weight decay on matrices, not on the sub-keys),
    SparseAdam for the bank's values when their gradient is sparse."""
    bank = getattr(model, "bank", None)
    sparse = [bank.values.weight] if bank is not None and bank.values.sparse else []
    skip = {id(p) for p in sparse}
    no_decay = {id(bank.subkeys)} if bank is not None else set()
    dense = [p for p in model.parameters() if id(p) not in skip]
    decay = [p for p in dense if p.ndim >= 2 and id(p) not in no_decay]
    rest = [p for p in dense if not (p.ndim >= 2 and id(p) not in no_decay)]
    opts = [torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                               {"params": rest, "weight_decay": 0.0}], lr=args.lr)]
    if sparse:
        opts.append(torch.optim.SparseAdam(sparse, lr=args.lr_bank))

    def shape(step):
        warm = min(1.0, (step + 1) / max(1, args.warmup))
        return warm * 0.5 * (1.0 + math.cos(math.pi * step / args.steps))

    scheds = [torch.optim.lr_scheduler.LambdaLR(o, shape) for o in opts]
    return opts, scheds


@torch.no_grad()
def correct(model, x, y, batch=4096):
    model.eval()
    hits = [model(x[i:i + batch]).argmax(dim=-1) == y[i:i + batch]
            for i in range(0, len(x), batch)]
    return torch.cat(hits).float()


def evaluate(model, task, max_train=16384):
    """Accuracies and, for the machine, the bank's use over every lookup of the pass."""
    bank = getattr(model, "bank", None)
    if bank is not None:
        bank.reset_counts()
        bank.counting = True
    same = task["test_x"] is task["train_x"]
    res = {"train_acc": correct(model, task["train_x"][:max_train],
                                task["train_y"][:max_train]).mean().item()}
    if same and len(task["train_x"]) <= max_train:
        res["test_acc"] = res["train_acc"]
    else:
        ok = correct(model, task["test_x"], task["test_y"])
        res["test_acc"] = ok.mean().item()
        if "test_group" in task:
            groups = task["test_group"]
            res["test_by_group"] = {str(int(v)): ok[groups == v].mean().item()
                                    for v in groups.unique()}
    if bank is not None:
        bank.counting = False
        heads = use_stats(bank.mass, bank.hits)
        res["bank_heads"] = heads
        res["bank"] = {k: sum(h[k] for h in heads) / len(heads) for k in heads[0]}
        res["bank_train_used"] = (bank.train_hits > 0).float().mean().item()
    return res


def train(model, task, args):
    opts, scheds = optimisers(model, args)
    bank = getattr(model, "bank", None)
    if bank is not None:
        bank.track_read = args.vdecay > 0
    x, y = task["train_x"], task["train_y"]
    n = len(x)
    order, at = torch.randperm(n), 0
    eval_at = {round(args.steps * (i + 1) / args.evals) for i in range(args.evals - 1)}
    curve = []
    for step in range(1, args.steps + 1):
        if at + args.batch > n:
            order, at = torch.randperm(n), 0
        idx = order[at:at + args.batch]
        at += args.batch
        model.train()
        if args.tick_loss:
            loss = walk_loss(model, x[idx], task["meta"])
        else:
            loss = F.cross_entropy(model(x[idx]), y[idx])
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        if bank is not None and args.vdecay > 0:
            bank.decay_read(args.vdecay * scheds[0].get_last_lr()[0])
        for o, s in zip(opts, scheds):
            o.step()
            s.step()
        if step in eval_at:
            res = evaluate(model, task, max_train=4096)
            curve.append([step, round(res["train_acc"], 4), round(res["test_acc"], 4)])
            print(f"  step {step} loss {loss.item():.4f} train {res['train_acc']:.3f} "
                  f"test {res['test_acc']:.3f}", flush=True)
    return curve


def run_id(args):
    skip = {"out", "label", "force", "threads"}
    text = json.dumps({k: v for k, v in sorted(vars(args).items()) if k not in skip})
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def already_done(path, rid):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return any(json.loads(line)["id"] == rid for line in f if line.strip())


def main(argv=None):
    args = parse(argv)
    rid = run_id(args)
    if not args.force and already_done(args.out, rid):
        print(f"skip {args.label or args.model} seed {args.seed}: already in {args.out}")
        return
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    task = make_task(args.task, **vars(args))
    model, ref = build(args, task)
    row = {"id": rid, "label": args.label or args.model, "task": args.task,
           "model": args.model, "seed": args.seed, "params": count_params(model),
           "madds": count_madds(model, task["length"]),
           "madds_parts": model.madds(task["length"]), "ref": ref,
           "width": getattr(model, "width", None), "chance": task["chance"],
           "n_train": len(task["train_x"]), "n_test": len(task["test_x"])}
    print(f"{row['label']} seed {args.seed}: params {row['params']} madds {row['madds']}",
          flush=True)
    start = time.time()
    row["curve"] = train(model, task, args)
    row.update(evaluate(model, task))
    if args.task == "chase" and args.model == "machine":
        row.update(look_inside(model, task))
    row["secs"] = round(time.time() - start, 1)
    row["args"] = vars(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  done: train {row['train_acc']:.3f} test {row['test_acc']:.3f} "
          f"bank {row.get('bank')} ({row['secs']}s)", flush=True)


if __name__ == "__main__":
    main()
