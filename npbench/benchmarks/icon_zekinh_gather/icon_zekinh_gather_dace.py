import numpy as np
import dace as dc

NB, NLEV, NPROMA = (dc.symbol(s, dtype=dc.int64) for s in ('NB', 'NLEV', 'NPROMA'))


@dc.program
def kernel(e_bln: dc.float32[NB, 3, NPROMA], edge_idx: dc.int32[NB, NPROMA, 3], edge_blk: dc.int32[NB, NPROMA, 3],
           z_kin_hor_e: dc.float32[NB, NLEV, NPROMA], z_ekinh: dc.float32[NB, NLEV, NPROMA]):
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                z_ekinh[jb, jk, jc] = (
                    e_bln[jb, 0, jc] * z_kin_hor_e[edge_blk[jb, jc, 0], jk, edge_idx[jb, jc, 0]] +
                    e_bln[jb, 1, jc] * z_kin_hor_e[edge_blk[jb, jc, 1], jk, edge_idx[jb, jc, 1]] +
                    e_bln[jb, 2, jc] * z_kin_hor_e[edge_blk[jb, jc, 2], jk, edge_idx[jb, jc, 2]])
