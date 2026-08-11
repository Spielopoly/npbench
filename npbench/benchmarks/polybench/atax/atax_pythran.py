import numpy as np


# pythran export kernel(float32[:,:], float32[:])
def kernel(A, x):

    return (A @ x) @ A
