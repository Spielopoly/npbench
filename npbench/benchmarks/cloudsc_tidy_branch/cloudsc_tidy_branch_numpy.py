import numpy as np

PTSPHY = 50.0
RLMIN = 1.0e-8
RAMIN = 1.0e-8
RALVDCP = 2.5008e6 / 1004.7
RALSDCP = 2.8345e6 / 1004.7
ZQTMST = 1.0 / PTSPHY


def kernel(zqx_l, zqx_i, zqx_v, za, ptend_q, ptend_t):
    cond = (zqx_l + zqx_i < RLMIN) | (za < RAMIN)
    zqadj_l = zqx_l[cond] * ZQTMST
    ptend_q[cond] += zqadj_l
    ptend_t[cond] -= RALVDCP * zqadj_l
    zqx_v[cond] += zqx_l[cond]
    zqx_l[cond] = 0.0
    zqadj_i = zqx_i[cond] * ZQTMST
    ptend_q[cond] += zqadj_i
    ptend_t[cond] -= RALSDCP * zqadj_i
    zqx_v[cond] += zqx_i[cond]
    zqx_i[cond] = 0.0
    za[cond] = 0.0
