from enum import IntEnum
from functools import lru_cache

from typing import TypeVar

import numpy as np


def euclidean_distance(coord1, coord2):
    return abs(np.linalg.norm(np.asarray(coord1) - coord2))


class RelationType(IntEnum):
    PRESYNAPTIC = 0
    POSTSYNAPTIC = 1

    @classmethod
    @lru_cache(1)
    def synaptic(cls):
        return {cls.PRESYNAPTIC, cls.POSTSYNAPTIC}


ValidRelType = TypeVar("ValidRelType", int, RelationType)
