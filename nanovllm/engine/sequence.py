from ast import List
from copy import copy
from enum import Enum, auto
from itertools import count


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    # start from 0
    counter = count()

    def __init__(self, token_ids: List[int]):
        self.id = next(Sequence.counter)
        self.token_ids = copy(token_ids)
        self.status = SequenceStatus.WAITING
        self.num_tokens = len(self.token_ids)
        self.block_table = []  # paged attention
        self.cached_tokens = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, index: int):
        return self.token_ids[index]


if __name__ == "__main__":
    sequence_status = SequenceStatus.WAITING
    print(sequence_status.name)
    print(sequence_status.value)
