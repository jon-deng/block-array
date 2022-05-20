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
        vec = reduce(
            lambda a, b: a+b,
            [gops.mult_mat_vec(submat, subvec) for submat, subvec in zip(submat_row, vec)])
        ret_subvecs.append(vec)
    return bv.BlockVector(ret_subvecs, labels=mat.labels[0:1])

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
    NROW, NCOL = mat_a.shape[0], mat_b.shape[1]

    assert mat_a.bshape[1] == mat_b.bshape[0]
    NREDUCE = mat_a.shape[1]

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

    labels = tuple([mat_a.labels[0], mat_b.labels[1]])
    return bm.BlockMatrix(mats, labels=labels)

