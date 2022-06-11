"""
This module contains various linear algebra operations on block matrices/vectors
"""

from functools import reduce
import typing

from . import subops as gops
from . import blockvec as bv
from . import blockmat as bm

# make norm and inner product for vectors availabel from blockvec.py
norm = bv.norm
dot = bv.dot

T = typing.TypeVar('T')

def mult_mat_vec(mat: bm.BlockMatrix[T], vec: bv.BlockVector[T]) -> bv.BlockVector[T]:
    """
    Return the block matrix-vector product

    Parameters
    ----------
    bmat : bm.BlockMatrix
        The block matrix
    bvec : bv.BlockVector
        The block vector

    Returns
    -------
    bv.BlockVector
        The resulting block vector from the matrix-vector product
    """
    ret_subvecs = []
    for submat_row in mat:
        ret_subvec = reduce(
            lambda a, b: a+b,
            [gops.mult_mat_vec(submat, subvec) for submat, subvec in zip(submat_row, vec)])
        ret_subvecs.append(ret_subvec)
    return bv.BlockVector(ret_subvecs, labels=mat.f_labels[0:1])

def mult_mat_mat(mat_a: bm.BlockMatrix[T], mat_b: bm.BlockMatrix[T]) -> bm.BlockMatrix[T]:
    """
    Return the block matrix-matrix product

    Parameters
    ----------
    mat_a, mat_b : bm.BlockMatrix
        The block matrices to multiply. This is done the order mat_a*mat_b

    Returns
    -------
    bm.BlockMatrix
        The resulting block matrix from the matrix-matrix product
    """
    ## ii/jj denote the current row/col indices
    NROW, NCOL = mat_a.f_shape[0], mat_b.f_shape[1]

    assert mat_a.f_bshape[1] == mat_b.f_bshape[0]
    NREDUCE = mat_a.f_shape[1]

    mats = []
    for ii in range(NROW):
        mat_row = [
            reduce(
                lambda a, b: a + b,
                [gops.mult_mat_mat(mat_a[ii, kk], mat_b[kk, jj]) for kk in range(NREDUCE)]
            )
            for jj in range(NCOL)
        ]
        mats.append(mat_row)

    labels = tuple([mat_a.f_labels[0], mat_b.f_labels[1]])
    return bm.BlockMatrix(mats, labels=labels)

