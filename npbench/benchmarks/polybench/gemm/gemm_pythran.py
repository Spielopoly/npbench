import numpy as np


# pythran export kernel(float32, float32, float32[:,:], float32[:,:], float32[:,:])
def kernel(alpha, beta, C, A, B):

    C[:] = alpha * A @ B + beta * C
