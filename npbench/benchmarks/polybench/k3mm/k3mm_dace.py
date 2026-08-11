import numpy as np
import dace as dc

NI, NJ, NK, NL, NM = (dc.symbol(s, dtype=dc.int64)
                      for s in ('NI', 'NJ', 'NK', 'NL', 'NM'))


@dc.program
def kernel(A: dc.float32[NI, NK], B: dc.float32[NK, NJ], C: dc.float32[NJ, NM],
           D: dc.float32[NM, NL]):

    return A @ B @ C @ D
