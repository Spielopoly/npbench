import numpy as np
import dace as dc

KLEV, KLON = (dc.symbol(s, dtype=dc.int64) for s in ('KLEV', 'KLON'))

PTSPHY = 50.0
RLMIN = 1.0e-8
RAMIN = 1.0e-8
RALVDCP = 2.5008e6 / 1004.7
RALSDCP = 2.8345e6 / 1004.7
ZQTMST = 1.0 / PTSPHY


@dc.program
def kernel(zqx_l: dc.float64[KLEV, KLON], zqx_i: dc.float64[KLEV, KLON], zqx_v: dc.float64[KLEV, KLON],
           za: dc.float64[KLEV, KLON], ptend_q: dc.float64[KLEV, KLON], ptend_t: dc.float64[KLEV, KLON]):
    for jk in range(KLEV):
        for jl in range(KLON):
            if zqx_l[jk, jl] + zqx_i[jk, jl] < RLMIN or za[jk, jl] < RAMIN:
                zqadj_l = zqx_l[jk, jl] * ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_l
                ptend_t[jk, jl] = ptend_t[jk, jl] - RALVDCP * zqadj_l
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_l[jk, jl]
                zqx_l[jk, jl] = 0.0
                zqadj_i = zqx_i[jk, jl] * ZQTMST
                ptend_q[jk, jl] = ptend_q[jk, jl] + zqadj_i
                ptend_t[jk, jl] = ptend_t[jk, jl] - RALSDCP * zqadj_i
                zqx_v[jk, jl] = zqx_v[jk, jl] + zqx_i[jk, jl]
                zqx_i[jk, jl] = 0.0
                za[jk, jl] = 0.0
