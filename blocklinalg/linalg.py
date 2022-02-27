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

def generic_mult_mat_vec(A, x):
    """
    Return a generic matrix vector multiplication
    """
    np_array_types = (np.ndarray, jax.numpy.ndarray)
    if isinstance(A, np_array_types) and isinstance(x, np_array_types):
        return A@x
    else:
        try:
            return A*x
        except:
            raise

def mult_mat_vec(A, x):
    y_vecs = []
    for submat_row in A.mats:
        y_vec = reduce(
            lambda a, b: a+b, 
            [generic_mult_mat_vec(submat, subvec) for submat, subvec in zip(submat_row, x.vecs)])
        y_vecs.append(y_vec)
    return BlockVec(y_vecs, x.keys)

# def mult_mat_mat(A, B):
#     A_v = []
#     for m_row, xvec in enumerate(x.vecs):
#         yvec = generic_mult_mat_mat(A.mats[m_row][0], vec)
#         for n in range(1, len(A.col_keys)):
#             yvec += generic_mult_mat_vec(A.mats[m_row][n], vec)
#         yvecs.append(yvec)
#     return BlockVec(yvecs, x.keys)

# def mult_mat_mat(A, B):