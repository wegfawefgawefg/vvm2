"""Checks of the parts that are easy to get silently wrong. Run by scripts/smoke.sh.

  python -m vvm2.check
"""
import torch

from .bank import Bank
from .baselines import GRU, MLP, Transformer, count_madds, count_params, fit_width
from .vvm import VVM


def check_exact_topk():
    """The product-key pick equals a brute-force top k over all N entries."""
    torch.manual_seed(0)
    bank = Bank(in_dim=24, entries=1024, d_value=8, heads=3, k=8, d_key=16).eval()
    s1, s2 = bank.scores(torch.randn(50, 24))
    idx, score = bank.pick(s1, s2)
    full = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(-2)  # every entry's score
    want = full.topk(8, dim=-1)
    assert torch.equal(idx, want.indices)
    assert torch.allclose(score, want.values, atol=1e-6)


def check_sparse_gradient():
    """Only the rows that were read get a gradient, and the query and sub-keys get one."""
    torch.manual_seed(0)
    bank = Bank(in_dim=24, entries=1024, d_value=8, heads=2, k=4, d_key=16)
    bank(torch.randn(5, 24)).sum().backward()
    grad = bank.values.weight.grad
    assert grad.is_sparse
    rows = grad.coalesce().indices()[0].unique()
    read = (bank.train_hits.sum(dim=0) > 0).nonzero().squeeze(1)
    assert torch.equal(rows, read) and len(rows) <= 5 * 2 * 4
    assert bank.query.weight.grad.abs().sum() > 0 and bank.subkeys.grad.abs().sum() > 0


def check_noise_is_training_only():
    torch.manual_seed(0)
    bank = Bank(in_dim=24, entries=1024, d_value=8, k=4, d_key=16, pick_from=32, gumbel=1.0,
                qnoise=0.5)
    x = torch.randn(6, 24)
    bank.eval()
    assert torch.equal(bank(x), bank(x))
    bank.train()
    assert not torch.equal(bank(x), bank(x))
    assert bank(x).shape == (6, 2, 8)


def check_accounting():
    """The vvm's compute does not count the unread bank: a bank 1024 times larger
    costs only the extra sub-key scoring (which grows with sqrt(N)). Baselines size to a
    target."""
    small = VVM(40, 10, entries=1024, ticks=2, heads=2, d_key=32)
    large = VVM(40, 10, entries=1024 * 1024, ticks=2, heads=2, d_key=32)
    assert count_params(large) - count_params(small) > 1000 * 1024 * 64
    extra = count_madds(large, 2) - count_madds(small, 2)
    assert extra == 2 * 2 * 32 * (1024 - 32)  # ticks * heads * d_key * (side difference)
    target = 200_000
    for make in (lambda w: MLP(40, 2, 10, w), lambda w: GRU(40, 2, 10, w),
                 lambda w: Transformer(40, 2, 10, w)):
        for measure in (count_params, lambda m: count_madds(m, 2)):
            got = measure(make(fit_width(make, measure, target, step=4)))
            assert abs(got - target) < 0.15 * target, (got, target)


def check_vvm_runs():
    x = torch.randint(40, (7, 5))
    for kw in (dict(), dict(regs=3, heads=4), dict(feed="seq", use_mem=True, regs=2),
               dict(use_bank=False), dict(use_alu=False), dict(hold=True, snoise=0.1)):
        record = []
        vvm = VVM(40, 10, entries=256, ticks=3, **kw)
        out = vvm(x, record=record)
        assert out.shape == (7, 10)
        assert len(record) == 3 * (5 if kw.get("feed") == "seq" else 1)
        out.sum().backward()
        assert count_madds(vvm, 5) > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("check_"):
            fn()
            print("ok", name)
