"""
This module contains generic functions for operating on vector/matrix objects 
from PETSc, numpy, and FEniCS.

This is needed for BlockVector and BlockMatrix to work with different
'subelements' from the different packages.
"""

import dolfin as dfn
import numpy as np
import jax
from petsc4py import PETSc

NDARRAY_LIKE_TYPES = (np.ndarray, jax.numpy.ndarray)

# pylint: disable=no-member


def set_vec(veca, vecb):
    """
    Set the specified values to a vector

    This handles the different indexing notations that vectors can have based on the type, shape,
    etc.

    Parameters
    ----------
    veca, vecb : dolfin.PETScVector, PETSc.Vec, np.ndarray
    """
    if isinstance(veca, NDARRAY_LIKE_TYPES) and veca.shape == ():
        veca[()] = vecb
    elif isinstance(veca, PETSc.Vec):
        veca.array[:] = vecb
    else:
        veca[:] = vecb

# TODO: This function probably doesn't work
def set_mat(mata, matb):
    """
    Set the specified values to a matrix

    This handles the different indexing notations that vectors can have based on the type, shape,
    etc.

    Parameters
    ----------
    mata, matb : dolfin.PETScMatrix, PETSc.Mat, np.ndarray
    """
    if isinstance(mata, NDARRAY_LIKE_TYPES):
        mata[:] = matb
    else:
        mata.set(matb)


def add_vec(veca, vecb, out=None):
    return veca+vecb

def add_mat(mata, matb, out=None):
    return mata+matb


def mult_mat_vec(mat, vec, out=None):
    """
    Return a matrix-vector product

    Parameters
    ----------
    mat : dolfin.PETScMatrix, PETSc.Mat, np.ndarray
    vec : dolfin.PETScVector, PETsc.Vec, np.ndarray
    """
    if isinstance(mat, NDARRAY_LIKE_TYPES) and isinstance(vec, NDARRAY_LIKE_TYPES):
        if out is None:
            out = mat@vec
        else:
            out[:] = mat@vec # TODO: This doesn't make use of out
    elif isinstance(mat, PETSc.Mat):
        out = mat.createVecLeft() if out is None else out
        mat.mult(vec, out)
    else:
        out = mat*vec
    return out

def mult_mat_mat(mata, matb, out=None):
    """
    Return a matrix-matrix product

    Parameters
    ----------
    mata, matb : dolfin.PETScMatrix, PETSc.Mat, np.ndarray
    """
    
    if isinstance(mata, NDARRAY_LIKE_TYPES) and isinstance(matb, NDARRAY_LIKE_TYPES):
        return mata@matb
    else:
        return mata.matMult(matb)


def norm_mat(mat):
    if isinstance(mat, PETSc.Mat):
        return mat.norm(norm_type=PETSc.NormType.FROBENIUS)
    else:
        return np.sqrt(np.sum(mat**2))


def size_vec(vec):
    """
    Return vector size

    Parameters
    ----------
    vec : dolfin.PETScVector, PETSc.Mat or np.ndarray
    """
    if isinstance(vec, NDARRAY_LIKE_TYPES):
        return vec.size
    elif isinstance(vec, PETSc.Vec):
        return vec.size
    else:
        return len(vec)

def shape_vec(vec):
    """
    Return vector shape

    This is just a tuple version of the `size_vec`

    Parameters
    ----------
    vec : dolfin.PETScVector, PETSc.Mat or np.ndarray
    """
    return (size_vec(vec),)

def size_mat(mat):
    """
    Return matrix size for different matrix types

    This is just the total number of elements in the matrix

    Parameters
    ----------
    generic_mat : PETSc.Mat or np.ndarray
    """
    shape = shape_mat(mat)
    return shape[0]*shape[1]

def shape_mat(mat):
    """
    Return matrix shape for different matrix types

    Parameters
    ----------
    generic_mat : PETSc.Mat or np.ndarray
    """
    if isinstance(mat, PETSc.Mat):
        return mat.getSize()
    if isinstance(mat, dfn.PETScMatrix):
        return tuple([mat.size(i) for i in range(2)])
    else:
        assert len(mat.shape) == 2
        return mat.shape


## Convert matrix/vector types
def convert_mat_to_petsc(mat, comm=None, keep_diagonal=True):
    """
    Return a `PETSc.Mat` representation of `mat`
    """
    mat_shape = shape_mat(mat)
    assert len(mat_shape) == 2
    if isinstance(mat, NDARRAY_LIKE_TYPES):
        COL_IDXS = np.arange(mat_shape[1], dtype=np.int32)
        out = PETSc.Mat().createAIJ(mat_shape, comm=comm)
        out.setUp()
        for ii in range(mat_shape[0]):
            current_row = np.array(mat[ii, :]).copy()
            idx_nonzero = np.array(current_row != 0).copy()
            out.setValues(
                ii, COL_IDXS[idx_nonzero], current_row[idx_nonzero], 
                addv=PETSc.InsertMode.ADD_VALUES)

        if keep_diagonal:
            for ii in range(mat_shape[0]):
                out.setValue(ii, ii, 0, addv=PETSc.InsertMode.ADD_VALUES)
        out.assemble()
    elif isinstance(mat, dfn.PETScMatrix):
        out = mat.mat()
    else:
        out = mat

    return out

def convert_vec_to_petsc(vec, comm=None):
    """
    Return a `PETSc.Vec` representation of `vec`
    """
    M = size_vec(vec)
    if isinstance(vec, NDARRAY_LIKE_TYPES):
        out = PETSc.Vec().createSeq(M, comm=comm)
        out.setUp()
        out.array[:] = vec
        out.assemble()
    elif isinstance(vec, dfn.PETScVector):
        out = vec.vec()
    else:
        out = vec
    return out
