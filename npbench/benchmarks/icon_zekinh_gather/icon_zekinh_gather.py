# ICON (velocity_zekinh_block.f90) bilinear cell-from-edges interpolation —
# 3-edge data-dependent gather weighted by per-cell coefficients.

import numpy as np


def initialize(NB, NLEV, NPROMA, datatype=np.float64):
    rng = np.random.default_rng(42)
    e_bln = rng.standard_normal((NB, 3, NPROMA)).astype(datatype)
    edge_idx = rng.integers(0, NPROMA, size=(NB, NPROMA, 3)).astype(np.int32)
    edge_blk = rng.integers(0, NB, size=(NB, NPROMA, 3)).astype(np.int32)
    z_kin_hor_e = rng.standard_normal((NB, NLEV, NPROMA)).astype(datatype)
    z_ekinh = np.zeros((NB, NLEV, NPROMA), dtype=datatype)

    return e_bln, edge_idx, edge_blk, z_kin_hor_e, z_ekinh
