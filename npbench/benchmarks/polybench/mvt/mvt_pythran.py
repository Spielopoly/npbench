import numpy as np


# pythran export kernel(float32[:], float32[:], float32[:], float32[:], float32[:,:])
def kernel(x1, x2, y_1, y_2, A):

    x1 += A @ y_1
    x2 += y_2 @ A
