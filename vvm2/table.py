"""Print results tables (markdown) from the rows that vvm2.train wrote.

  python -m vvm2.table out/0b_assoc.jsonl --split items
  python -m vvm2.table out/0b_assoc.jsonl --pivot items   test accuracy, one column a value
  python -m vvm2.table out/0d_chase.jsonl --groups        accuracy per test group (h)
  python -m vvm2.table out/0d_chase.jsonl --lens "machine T=4"   the look inside

Rows with the same label (and the same --split setting) are seeds of one line. Every line
has the parameter count and the multiply-adds per example. Accuracies are the mean over
seeds with the lowest and highest seed in brackets. Bank columns are the mean over heads
and seeds: used = share of entries picked at all, ent = entropy of the read weight as a
share of log N, eff = exp(entropy) = effective number of entries, top = the top entry's
share of the read weight, trained = share of entries ever picked in training.
"""
import argparse
import json
from collections import defaultdict


def load(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def short(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(int(n))


def mean(xs):
    return sum(xs) / len(xs)


def spread(xs):
    return f"{mean(xs):.3f} [{min(xs):.3f}, {max(xs):.3f}]"


def grouped(rows, split):
    """{split value: {label: [rows]}}; machine lines first, then by label."""
    out = defaultdict(lambda: defaultdict(list))
    for r in sorted(rows, key=lambda r: (r["model"] != "machine", r["label"], r["seed"])):
        out[r["args"].get(split) if split else None][r["label"]].append(r)
    return dict(sorted(out.items(), key=lambda kv: (isinstance(kv[0], str), kv[0] or 0)))


def pivot_table(rows, by):
    """One line per label, test accuracy (mean over seeds) at each value of a setting."""
    lines, values = defaultdict(dict), sorted({r["args"][by] for r in rows})
    for value, labels in grouped(rows, by).items():
        for label, rs in labels.items():
            lines[label]["sizes"] = [short(mean([r["params"] for r in rs])),
                                     short(mean([r["madds"] for r in rs]))]
            lines[label][value] = f"{mean([r['test_acc'] for r in rs]):.3f}"
    order = grouped(rows, "")[None]
    print_table(["model", "params", "madds"] + [f"{by} = {v}" for v in values],
                [[label] + lines[label]["sizes"] + [lines[label].get(v, "-") for v in values]
                 for label in order])


def print_table(header, lines):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for line in lines:
        print("| " + " | ".join(str(c) for c in line) + " |")
    print()


def main_table(rows):
    header = ["model", "params", "madds", "train acc", "test acc", "used", "ent", "eff",
              "top", "trained", "seeds"]
    lines = []
    for label, rs in rows.items():
        line = [label, short(mean([r["params"] for r in rs])),
                short(mean([r["madds"] for r in rs])),
                spread([r["train_acc"] for r in rs]), spread([r["test_acc"] for r in rs])]
        banks = [r["bank"] for r in rs if "bank" in r]
        if banks:
            line += [f"{mean([b['used'] for b in banks]):.3f}",
                     f"{mean([b['ent'] for b in banks]):.3f}",
                     f"{mean([b['eff'] for b in banks]):.0f}",
                     f"{mean([b['top'] for b in banks]):.4f}",
                     f"{mean([r['bank_train_used'] for r in rs]):.3f}"]
        else:
            line += ["-"] * 5
        lines.append(line + [len(rs)])
    print_table(header, lines)


def group_table(rows):
    keys = sorted({k for rs in rows.values() for r in rs for k in r.get("test_by_group", {})},
                  key=int)
    lines = []
    for label, rs in rows.items():
        rs = [r for r in rs if "test_by_group" in r]
        if rs:
            lines.append([label, short(mean([r["params"] for r in rs])),
                          short(mean([r["madds"] for r in rs]))]
                         + [f"{mean([r['test_by_group'].get(k, float('nan')) for r in rs]):.3f}"
                            for k in keys])
    print_table(["model", "params", "madds"] + [f"test acc, group {k}" for k in keys], lines)


def head_table(rows):
    lines = []
    for label, rs in rows.items():
        for h in range(len(rs[0].get("bank_heads", []))):
            heads = [r["bank_heads"][h] for r in rs]
            lines.append([label, h] + [f"{mean([x[k] for x in heads]):.4f}"
                                       for k in ("used", "ent", "eff", "top")])
    print_table(["model", "head", "used", "ent", "eff", "top"], lines)


def lens_table(rows, label):
    rs = [r for r in rows.get(label, []) if "lens" in r]
    if not rs:
        print(f"no lens rows for {label!r}\n")
        return
    for count in sorted(rs[0]["lens"], key=int):
        mats = [r["lens"][count] for r in rs]
        hops = len(mats[0][0])
        print(f"{label}, inputs with h = {count}: share whose state after the tick reads "
              f"out as f applied i times to x (mean of {len(rs)} seeds)\n")
        lines = [[t + 1] + [f"{mean([m[t][i] for m in mats]):.3f}" for i in range(hops)]
                 for t in range(len(mats[0]))]
        print_table(["tick"] + [f"i = {i}" for i in range(hops)], lines)
    probes = [r["probe"] for r in rs]
    print("fitted probe, held-out start items, accuracy per tick: "
          + ", ".join(f"{mean([p[t] for p in probes]):.3f}" for t in range(len(probes[0])))
          + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+")
    p.add_argument("--split", default="", help="a setting; one table per value of it")
    p.add_argument("--pivot", default="", help="a setting; test accuracy per value of it")
    p.add_argument("--groups", action="store_true", help="test accuracy per test group")
    p.add_argument("--heads", action="store_true", help="bank use per head")
    p.add_argument("--lens", default="", help="label of the machine rows to look inside")
    args = p.parse_args()
    if args.pivot:
        pivot_table(load(args.paths), args.pivot)
        return
    for value, rows in grouped(load(args.paths), args.split).items():
        if args.split:
            print(f"**{args.split} = {value}**\n")
        if args.lens:
            lens_table(rows, args.lens)
        elif args.groups:
            group_table(rows)
        elif args.heads:
            head_table(rows)
        else:
            main_table(rows)


if __name__ == "__main__":
    main()
