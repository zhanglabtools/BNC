from __future__ import annotations

from bnc_repro.data.modular_addition import split_modular_addition
from bnc_repro.protocols.common import geometric_checkpoint_epochs
from bnc_repro.protocols.rank2_finetune import should_record_s2
from bnc_repro.protocols.rank_homotopy import should_record_s1


def test_mod97_split_sizes() -> None:
    split = split_modular_addition(97, 0.7, 1)
    assert len(split.train_indices) == 6586
    assert len(split.test_indices) == 2823


def test_checkpoint_counts() -> None:
    assert len(range(0, 10001, 50)) == 201
    assert sum(should_record_s1(epoch, 10000) for epoch in range(10001)) == 247
    assert sum(should_record_s2(epoch, 6000) for epoch in range(6001)) == 561
    assert len(geometric_checkpoint_epochs(20000, 321)) == 240
    assert geometric_checkpoint_epochs(20000, 321)[0] == 1
    assert geometric_checkpoint_epochs(20000, 321)[-1] == 20000

