import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def kernel(x1: dc.float32[N], x2: dc.float32[N], y_1: dc.float32[N],
           y_2: dc.float32[N], A: dc.float32[N, N]):

    x1 += A @ y_1
    x2 += y_2 @ A
