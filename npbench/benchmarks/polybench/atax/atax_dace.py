import numpy as np
import dace as dc

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def kernel(A: dc.float32[M, N], x: dc.float32[N]):

    return (A @ x) @ A
