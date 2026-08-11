# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, tEnd, dt):
    from numpy.random import default_rng
    rng = default_rng(42)
    mass = np.full((N, 1), np.float32(20.0 / N), dtype=np.float32)
    pos = rng.random((N, 3), dtype=np.float32)
    vel = rng.random((N, 3), dtype=np.float32)
    Nt = int(np.ceil(tEnd / dt))
    return mass, pos, vel, Nt
