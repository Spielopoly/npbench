# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    t0 = rng.random((N, ), dtype=np.float32)
    p0 = rng.random((N, ), dtype=np.float32)
    t1 = rng.random((N, ), dtype=np.float32)
    p1 = rng.random((N, ), dtype=np.float32)
    return t0, p0, t1, p1
