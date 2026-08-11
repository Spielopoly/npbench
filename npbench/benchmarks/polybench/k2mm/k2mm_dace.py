import numpy as np
import dace as dc

NI, NJ, NK, NL = (dc.symbol(s, dtype=dc.int64)
                  for s in ('NI', 'NJ', 'NK', 'NL'))


@dc.program
def kernel(alpha: dc.float32, beta: dc.float32, A: dc.float32[NI, NK],
           B: dc.float32[NK, NJ], C: dc.float32[NJ, NL], D: dc.float32[NI,
                                                                       NL]):

    D[:] = alpha * A @ B @ C + beta * D
