import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def kernel(alpha: dc.float32, beta: dc.float32, A: dc.float32[N, N],
           B: dc.float32[N, N], x: dc.float32[N]):

    return alpha * A @ x + beta * B @ x
