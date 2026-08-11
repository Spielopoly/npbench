import numpy as np


# pythran export kernel(float32[:,:])
def kernel2(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)
