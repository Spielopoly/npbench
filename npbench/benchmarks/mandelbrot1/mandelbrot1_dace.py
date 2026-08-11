# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------

import numpy as np
import dace as dc

XN, YN, N = (dc.symbol(s, dtype=dc.int64) for s in ['XN', 'YN', 'N'])


@dc.program
def linspace(start: dc.float32, stop: dc.float32, X: dc.float32[N]):
    dist = (stop - start) / (N - 1)
    for i in dc.map[0:N]:
        X[i] = start + i * dist


@dc.program
def mandelbrot(xmin: dc.float32, xmax: dc.float32, ymin: dc.float32,
               ymax: dc.float32, maxiter: dc.int64, horizon: dc.float32):
    # Adapted from https://www.ibm.com/developerworks/community/blogs/jfp/...
    #              .../entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en
    X = np.ndarray((XN, ), dtype=np.float32)
    Y = np.ndarray((YN, ), dtype=np.float32)
    linspace(xmin, xmax, X)
    linspace(ymin, ymax, Y)
    # C = X + np.reshape(Y, (YN, 1)) * 1j
    C = np.ndarray((YN, XN), dtype=np.complex64)
    for i, j in dc.map[0:YN, 0:XN]:
        C[i, j] = X[j] + Y[i] * 1j
    N = np.zeros(C.shape, dtype=np.int64)
    Z = np.zeros(C.shape, dtype=np.complex64)
    for n in range(maxiter):
        I = np.less(np.absolute(Z), horizon)
        N[I] = n
        # np.positive(n, out=N, where=I)
        # for j, k in dace.map[0:YN, 0:XN]:
        #     if I[j, k]:
        #         N[j, k] = n
        # Z[I] = Z[I]**2 + C[I]
        # np.add(np.power(Z, 2, where=I), C, out=Z, where=I)
        for j, k in dc.map[0:YN, 0:XN]:
            if I[j, k]:
                Z[j, k] = Z[j, k]**2 + C[j, k]
    N[N == maxiter - 1] = 0
    # np.positive(0, out=N, where=N==maxiter-1)
    # for j, k in dace.map[0:YN, 0:XN]:
    #     if N[j, k] == maxiter-1:
    #         N[j, k] = 0
    return Z, N
