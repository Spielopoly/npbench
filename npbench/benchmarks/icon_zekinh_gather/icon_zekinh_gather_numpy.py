import numpy as np


def kernel(e_bln, edge_idx, edge_blk, z_kin_hor_e, z_ekinh):
    z_ekinh[:] = 0.0
    for e in range(3):
        blk = edge_blk[:, :, e]
        idx = edge_idx[:, :, e]
        # z_kin_hor_e[blk, :, idx] has shape (NB, NPROMA, NLEV)
        gathered = z_kin_hor_e[blk, :, idx].transpose(0, 2, 1)
        z_ekinh += e_bln[:, e, :][:, None, :] * gathered
