import numpy as np
import dace as dc

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def kernel(alpha: dc.float32, beta: dc.float32, A: dc.float32[N, N],
           u1: dc.float32[N], v1: dc.float32[N], u2: dc.float32[N],
           v2: dc.float32[N], w: dc.float32[N], x: dc.float32[N],
           y: dc.float32[N], z: dc.float32[N]):

    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
