import numpy as np
import dace as dc

NI, NJ, NK = (dc.symbol(s, dtype=dc.int64) for s in ('NI', 'NJ', 'NK'))


@dc.program
def kernel(alpha: dc.float32, beta: dc.float32, C: dc.float32[NI, NJ],
           A: dc.float32[NI, NK], B: dc.float32[NK, NJ]):

    C[:] = alpha * A @ B + beta * C
