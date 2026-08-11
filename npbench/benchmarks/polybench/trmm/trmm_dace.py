import numpy as np
import dace as dc

M, N, S = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N', 'S'))

# @dc.program
# def dot(l: dc.float32[S], r: dc.float32[S]):
#     return np.add.reduce(np.multiply(l, r))


@dc.program
def kernel(alpha: dc.float32, A: dc.float32[M, M], B: dc.float32[M, N]):

    for i in range(M):
        for j in range(N):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha
