# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    data = rng.random((N, ), dtype=np.float32)
    radius = rng.random((N, ), dtype=np.float32)
    return data, radius
