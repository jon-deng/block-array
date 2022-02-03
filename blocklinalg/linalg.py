"""
This module contains various linear algebra operations on block matrices/vectors
"""

import operator
from collections import OrderedDict

import numpy as np
from petsc4py import PETSc

# from .vec import BlockVec, general_vec_set, generic_vec_size
from .vec import *
from .mat import *

def generic_mult_mat_vec(A, x):
    """
    Return a generic matrix vector multiplication
    """
    if isinstance(A, PETSc.Mat):
        y = A.getVecLeft()
        return A.matMult(x, y)
    else:
        raise NotImplementedError("")

def mult_mat_vec(A, x):
    y_vecs = []
    for m_row, x_vec in enumerate(x.vecs):
        y_vec = generic_mult_mat_vec(A.mats[m_row][0], x_vec)
        for n in range(1, len(A.col_keys)):
            y_vec += generic_mult_mat_vec(A.mats[m_row][n], x_vec)
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