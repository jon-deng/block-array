"""
This module contains various linear algebra operations on block matrices/vectors
"""

from functools import reduce
import itertools
from multiprocessing.sharedctypes import Value
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
    ret_shape = (mat.f_shape[0],)
    ret_subvecs = []
    # Uncollapse any blocks so that iteration over rows and columns is done
    # The correct output shape (potentially collapsed), should be maintained
    # from `ret_shape`
    for submat_row in mat.unsqueeze():
        ret_subvec = reduce(
            lambda a, b: a+b,
            [
                gops.mult_mat_vec(submat, subvec)
                for submat, subvec in zip(submat_row.sub_blocks, vec)
            ]
        )
        ret_subvecs.append(ret_subvec)
    return bv.BlockVector(ret_subvecs, shape=ret_shape, labels=mat.f_labels[0:1])

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
    if mat_a.f_bshape[1] != mat_b.f_bshape[0]:
        raise ValueError(
            f"Matrices have incompatible block shapes {mat_a.f_bshape} and {mat_b.f_bshape}"
        )

    ret_shape = (mat_a.f_shape[0], mat_b.f_shape[1])
    ret_labels = tuple([mat_a.f_labels[0], mat_b.f_labels[1]])

    ## ii/jj denote the current row/col indices
    NROW, NCOL = mat_a.f_shape[0], mat_b.f_shape[1]

    assert mat_a.f_bshape[1] == mat_b.f_bshape[0]
    NREDUCE = mat_a.f_shape[1]

    ret_mats = [
        reduce(
            lambda a, b: a+b,
            [
                gops.mult_mat_mat(mat_a.sub_blocks[ii, kk], mat_b.sub_blocks[kk, jj])
                for kk in range(NREDUCE)
            ]
        )
        for ii, jj in itertools.product(range(NROW), range(NCOL))
    ]
    return bm.BlockMatrix(ret_mats, shape=ret_shape, labels=ret_labels)

