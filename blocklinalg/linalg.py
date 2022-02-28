"""
This module contains various linear algebra operations on block matrices/vectors
"""

import operator
from collections import OrderedDict
from functools import reduce

import numpy as np
import jax
import dolfin as dfn
from petsc4py import PETSc

# from .vec import BlockVec, general_vec_set, generic_vec_size
from .vec import *
from .mat import *

from . import genericops as gops

def mult_mat_vec(bmat, bvec):
    vecs = []
    for submat_row in bmat.mats:
        vec = reduce(
            lambda a, b: a+b, 
            [gops.mult_mat_vec(submat, subvec) for submat, subvec in zip(submat_row, bvec.vecs)])
        vecs.append(vec)
    return BlockVec(vecs, bvec.keys)

def mult_mat_mat(bmata, bmatb):
    ## ii/jj denote the current row/col indices
    NROW, NCOL = bmata.bsize[0], bmatb.bsize[1]
    
    assert bmata.bsize[1] == bmatb.bsize[0]
    NREDUCE = bmata.bsize[1]

    mats = []
    for ii in range(NROW):
        mat_row = [
            reduce(
                lambda a, b: a + b, 
                [gops.mult_mat_mat(bmata.mats[ii][kk], bmatb.mats[kk][jj]) for kk in range(NREDUCE)]
            )
            for jj in range(NCOL)
        ]
        mats.append(mat_row)
    return BlockMat(mats, bmata.row_keys, bmatb.col_keys)
