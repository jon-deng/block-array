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

# from .vec import BlockVector, general_vec_set, generic_vec_size
from .vec import *
from .mat import *

from . import subops as gops

def mult_mat_vec(bmat, bvec):
    vecs = []
    for submat_row in bmat:
        vec = reduce(
            lambda a, b: a+b,
            [gops.mult_mat_vec(submat, subvec) for submat, subvec in zip(submat_row, bvec)])
        vecs.append(vec)
    return BlockVector(vecs, labels=bmat.labels[0:1])

def mult_mat_mat(bmata, bmatb):
    ## ii/jj denote the current row/col indices
    NROW, NCOL = bmata.shape[0], bmatb.shape[1]

    assert bmata.bshape[1] == bmatb.bshape[0]
    NREDUCE = bmata.shape[1]

    mats = []
    for ii in range(NROW):
        mat_row = [
            reduce(
                lambda a, b: a + b,
                [gops.mult_mat_mat(bmata[ii, kk], bmatb[kk, jj]) for kk in range(NREDUCE)]
            )
            for jj in range(NCOL)
        ]
        mats.append(mat_row)

    labels = tuple([bmata.labels[0], bmatb.labels[1]])
    return BlockMatrix(mats, labels=labels)
