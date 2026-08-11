import numpy as np


# pythran export kernel(float32, float32, float32[:,:], float32[:,:], float32[:,:], float32[:,:])
def kernel(alpha, beta, A, B, C, D):

    D[:] = alpha * A @ B @ C + beta * D
