# CLOUDSC (cloudsc_bottom_lower.F90) "tidy up very small cloud cover or total
# cloud water" — guarded read-modify-write over several arrays.

import numpy as np


def initialize(KLEV, KLON, datatype=np.float32):
    rng = np.random.default_rng(42)
    zqx_l = rng.standard_normal((KLEV, KLON)).astype(datatype) * 1e-4
    zqx_i = rng.standard_normal((KLEV, KLON)).astype(datatype) * 1e-4
    # Make ~half the lanes tiny so both arms of the guard are exercised.
    tiny = rng.random((KLEV, KLON)) < 0.5
    zqx_l[tiny] = 1e-12
    zqx_i[tiny] = 1e-12
    za = rng.random((KLEV, KLON)).astype(datatype)
    za[rng.random((KLEV, KLON)) < 0.3] = 1e-12
    zqx_v = rng.standard_normal((KLEV, KLON)).astype(datatype)
    ptend_q = rng.standard_normal((KLEV, KLON)).astype(datatype)
    ptend_t = rng.standard_normal((KLEV, KLON)).astype(datatype)

    return zqx_l, zqx_i, zqx_v, za, ptend_q, ptend_t
