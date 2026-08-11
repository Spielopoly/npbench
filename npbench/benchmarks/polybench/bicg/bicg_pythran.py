import numpy as np


# pythran export kernel(float32[:,:], float32[:], float32[:])
def kernel(A, p, r):

    return r @ A, A @ p
