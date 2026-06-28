"""Unit test for the 16:4 open-set per-subject temporal window split.

Runnable either via pytest or directly:  python experiments/shared/test_openset_split.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from shared.pipeline import EEGPipeline  # noqa: E402


def _indices(n_cls=16, n_per=20, val_frac=0.15):
    y = torch.tensor([c for c in range(n_cls) for _ in range(n_per)], dtype=torch.long)
    return y, EEGPipeline._openset_indices(y, val_frac)


def test_openset_split_sizes_and_coverage():
    n_cls, n_per, val_frac = 16, 20, 0.15
    y, (train_idx, val_idx) = _indices(n_cls, n_per, val_frac)
    n_val = int(n_per * val_frac)        # 3
    n_train = n_per - n_val - 1          # 16 (one boundary-gap window dropped)
    assert len(train_idx) == n_cls * n_train, (len(train_idx), n_cls * n_train)
    assert len(val_idx) == n_cls * n_val, (len(val_idx), n_cls * n_val)
    # All subjects appear in BOTH splits (window-level, not subject-level).
    assert set(y[train_idx].tolist()) == set(range(n_cls))
    assert set(y[val_idx].tolist()) == set(range(n_cls))
    # Train/val indices are disjoint.
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0


def test_openset_split_temporal_gap_per_subject():
    n_cls, n_per, val_frac = 16, 20, 0.15
    _, (train_idx, val_idx) = _indices(n_cls, n_per, val_frac)
    tr, va = set(train_idx.tolist()), set(val_idx.tolist())
    for c in range(n_cls):
        base = c * n_per
        tr_local = sorted(i - base for i in tr if base <= i < base + n_per)
        va_local = sorted(i - base for i in va if base <= i < base + n_per)
        # Strict temporal cut plus a >=1-window boundary gap.
        assert max(tr_local) < min(va_local)
        assert min(va_local) - max(tr_local) >= 2


if __name__ == "__main__":
    test_openset_split_sizes_and_coverage()
    test_openset_split_temporal_gap_per_subject()
    print("OPENSET_SPLIT_OK")
