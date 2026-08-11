import numpy as np


# pythran export kernel(float32[:,:], float32[:,:], float32[:,:], float32[:,:])
def kernel(A, B, C, D):

    return A @ B @ C @ D
