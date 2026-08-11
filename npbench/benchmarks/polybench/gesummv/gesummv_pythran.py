import numpy as np


# pythran export kernel(float32, float32, float32[:,:], float32[:,:], float32[:])
def kernel(alpha, beta, A, B, x):

    return alpha * A @ x + beta * B @ x
