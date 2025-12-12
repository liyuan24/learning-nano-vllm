"""
pytest tests/test_block_manager.py -v
"""

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


def test_can_append():
    block_manager = BlockManager(1, 10)
    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    block_manager.allocate(seq)
    seq.append_token(11)
    assert not block_manager.can_append(seq)


def test_may_append():
    block_manager = BlockManager(2, 10)
    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    block_manager.allocate(seq)
    seq.append_token(11)
    assert block_manager.can_append(seq)
    block_manager.may_append(seq)
    assert len(seq.block_table) == 2
    assert seq.block_table[-1] == 1
