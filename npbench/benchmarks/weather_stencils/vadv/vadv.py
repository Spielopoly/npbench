# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(I, J, K):
    from numpy.random import default_rng
    rng = default_rng(42)

    dtr_stage = np.float32(3. / 20.)

    # Define arrays
    utens_stage = rng.random((I, J, K), dtype=np.float32)
    u_stage = rng.random((I, J, K), dtype=np.float32)
    wcon = rng.random((I + 1, J, K), dtype=np.float32)
    u_pos = rng.random((I, J, K), dtype=np.float32)
    utens = rng.random((I, J, K), dtype=np.float32)

    return dtr_stage, utens_stage, u_stage, wcon, u_pos, utens
