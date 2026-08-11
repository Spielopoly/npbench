import numpy as np
import dace as dc

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def kernel(A: dc.float32[N, M], p: dc.float32[M], r: dc.float32[N]):

    return r @ A, A @ p
