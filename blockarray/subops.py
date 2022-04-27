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

# pylint: disable=no-member

NDARRAY_TYPES = (np.ndarray, jax.numpy.ndarray)
PETSC_VECTOR_TYPES = (dfn.PETScVector, PETSc.Vec)
PETSC_MATRIX_TYPES = (dfn.PETScMatrix, PETSc.Mat)
# VECTOR_TYPES = NDARRAY_LIKE_TYPES + PETSC_VECTOR_TYPES


def set_vec(veca, vecb):
    """
    Set the specified values to a vector

    This handles the different indexing notations that vectors can have based on the type, shape,
    etc.

    Parameters
    ----------
    veca, vecb : dolfin.PETScVector, PETSc.Vec, np.ndarray
    """
    if isinstance(veca, NDARRAY_TYPES) and veca.shape == ():
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
    if isinstance(mata, NDARRAY_TYPES):
        mata[:] = matb
    else:
        mata.set(matb)


def add_vec(veca, vecb, out=None):
    return veca+vecb

def add_mat(mata, matb, out=None):
    return mata+matb


def solve_petsc_lu(amat, b, out=None, ksp=None):
    """
    Solve Ax=b using PETSc's LU solver
    """
    if ksp is None:
        ksp = PETSc.KSP().create()
        ksp.setType(ksp.Type.PREONLY)
        ksp.setOperators(amat)
        ksp.setUp()

        pc = ksp.getPC()
        pc.setType(pc.Type.LU)

    if out is None:
        out = amat.getVecRight()
    ksp.solve(b, out)
    return out, ksp

def mult_mat_vec(mat, vec, out=None):
    """
    Return a matrix-vector product

    Parameters
    ----------
    mat : dolfin.PETScMatrix, PETSc.Mat, np.ndarray
    vec : dolfin.PETScVector, PETsc.Vec, np.ndarray
    """
    if isinstance(mat, NDARRAY_TYPES) and isinstance(vec, NDARRAY_TYPES):
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

    if isinstance(mata, NDARRAY_TYPES) and isinstance(matb, NDARRAY_TYPES):
        return mata@matb
    else:
        matc = PETSc.Mat().createAIJ([mata.size[1], 5])
        matc.setUp()
        matc.setValue(0, 0, 1)
        matc.assemble()

        matc = PETSc.Mat().createAIJ([matb.size[1], 5])
        matc.setUp()
        matc.setValue(0, 0, 1)
        matc.assemble()
        return mata.matMult(matb)

def norm_vec(vec):
    if isinstance(vec, PETSc.Vec):
        return vec.norm()
    elif isinstance(vec, dfn.PETScVector):
        return vec.norm('l2')
    else:
        return np.linalg.norm(vec)

def norm_mat(mat):
    if isinstance(mat, PETSc.Mat):
        return mat.norm(norm_type=PETSc.NormType.FROBENIUS)
    else:
        return np.sqrt(np.sum(mat**2))


def size(tensor):
    """
    Return the size of a tensor
    """
    if isinstance(tensor, PETSC_VECTOR_TYPES):
        return size_vec(tensor)
    elif isinstance(tensor, PETSC_MATRIX_TYPES):
        return size_mat(tensor)
    else:
        return tensor.size

def size_vec(vec):
    """
    Return vector size

    Parameters
    ----------
    vec : dolfin.PETScVector, PETSc.Mat or np.ndarray
    """
    if isinstance(vec, NDARRAY_TYPES):
        return vec.size
    elif isinstance(vec, PETSc.Vec):
        return vec.size
    else:
        return len(vec)

def shape(tensor):
    """
    Return the shape of a tensor
    """
    if isinstance(tensor, PETSC_VECTOR_TYPES):
        return shape_vec(tensor)
    elif isinstance(tensor, PETSC_MATRIX_TYPES):
        return shape_mat(tensor)
    else:
        return tensor.shape

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
    is_square = mat_shape[0] == mat_shape[1]
    if isinstance(mat, NDARRAY_TYPES):
        COL_IDXS = np.arange(mat_shape[1], dtype=np.int32)
        out = PETSc.Mat().createAIJ(mat_shape, comm=comm)
        out.setUp()
        for ii in range(mat_shape[0]):
            current_row = np.array(mat[ii, :])
            idx_nonzero = np.array(current_row != 0)

            rows = [ii]
            cols = COL_IDXS[idx_nonzero]
            vals = current_row[idx_nonzero]
            out.setValues(rows, cols, vals, addv=PETSc.InsertMode.ADD_VALUES)

        if keep_diagonal and is_square:
            for ii in range(mat_shape[0]):
                out.setValue(ii, ii, 0.0, addv=PETSc.InsertMode.ADD_VALUES)
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
    if isinstance(vec, NDARRAY_TYPES):
        out = PETSc.Vec().createSeq(M, comm=comm)
        out.setUp()
        out.array[:] = vec
        out.assemble()
    elif isinstance(vec, dfn.PETScVector):
        out = vec.vec()
    else:
        out = vec
    return out

## Convert vectors to row/column matrices
def convert_vec_to_rowmat(vec, comm=None):
    if isinstance(vec, dfn.PETScVector):
        vec = vec.vec()

    if isinstance(vec, PETSc.Vec):
        vec = np.array(vec.array)

    return convert_mat_to_petsc(vec.reshape(1, vec.size))

def convert_vec_to_colmat(vec, comm=None):
    if isinstance(vec, dfn.PETScVector):
        vec = vec.vec()

    if isinstance(vec, PETSc.Vec):
        vec = np.array(vec.array)

    return convert_mat_to_petsc(vec.reshape(vec.size, 1))
