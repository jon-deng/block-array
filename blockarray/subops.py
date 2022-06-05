"""
This module contains generic functions for operating on vector/matrix objects
from PETSc, numpy, and FEniCS.

This is needed for BlockVector and BlockMatrix to work with different
'subarrays' from the different packages.
"""

from multiprocessing.sharedctypes import Value
from typing import TypeVar, Union, Tuple, Optional
import math

import numpy as np
from . import _HAS_PETSC, _HAS_FENICS, _HAS_JAX, require_petsc, require_fenics
if _HAS_JAX:
    from jax import numpy as jnp
if _HAS_PETSC:
    from petsc4py import PETSc
if _HAS_FENICS:
    import dolfin as dfn

from .typing import (Shape, DfnMat, DfnVec, PETScMat, PETScVec, JaxArray)

# pylint: disable=no-member

NDARRAY_TYPES = (np.ndarray,) 
VECTOR_TYPES = ()
MATRIX_TYPES = ()
if _HAS_PETSC:
    VECTOR_TYPES += (PETScVec,)
    MATRIX_TYPES += (PETScMat,)
if _HAS_FENICS:
    VECTOR_TYPES += (DfnVec,)
    MATRIX_TYPES += (DfnMat,)
if _HAS_JAX:
    NDARRAY_TYPES += (JaxArray,)

# VECTOR_TYPES = NDARRAY_LIKE_TYPES + PETSC_VECTOR_TYPES

ALL_TYPES = NDARRAY_TYPES+VECTOR_TYPES+MATRIX_TYPES
ALL_VECTOR_TYPES = NDARRAY_TYPES+VECTOR_TYPES
ALL_MATRIX_TYPES = NDARRAY_TYPES+MATRIX_TYPES
if len(ALL_TYPES) == 1:
    T = TypeVar('T', bound=ALL_TYPES[0])
else:
    T = TypeVar('T', *ALL_TYPES)

if len(ALL_VECTOR_TYPES) == 1:
    V = TypeVar('V', bound=ALL_VECTOR_TYPES[0])
else:
    V = TypeVar('V', *ALL_VECTOR_TYPES)

if len(ALL_MATRIX_TYPES) == 1:
    M = TypeVar('M', bound=ALL_MATRIX_TYPES[0])
else:
    M = TypeVar('M', *ALL_MATRIX_TYPES)

## Core operations for computing size and shape of the various array types

def size(array: T) -> int:
    """
    Return the size of an array (total number of elements)

    Parameters
    ----------
    array : T

    Returns
    -------
    int
    """
    if isinstance(array, VECTOR_TYPES):
        return _size_vec(array)
    elif isinstance(array, MATRIX_TYPES):
        return _size_mat(array)
    elif isinstance(array, NDARRAY_TYPES):
        return array.size
    else:
        raise TypeError(f"Unknown size for array of type {type(array)}")

@require_fenics
@require_petsc
def _size_vec(vec: Union[DfnVec, PETScVec]) -> int:
    """
    Return the vector size (total number of elements)

    Parameters
    ----------
    vec : dolfin.PETScVector, PETScMat

    Returns
    -------
    int
    """
    if isinstance(vec, PETScVec):
        return vec.size
    elif isinstance(vec, DfnVec):
        return vec.size()
    else:
        raise TypeError(f"Unknown vector of type {type(vec)}")

@require_fenics
@require_petsc
def _size_mat(mat: Union[DfnMat, PETScMat]) -> int:
    """
    Return the matrix size (total number of elements)

    Parameters
    ----------
    mat : DfnMat, PETScMat

    Returns
    -------
    int
    """
    return math.prod(_shape_mat(mat))

def shape(array: T) -> Shape:
    """
    Return the shape of an array

    Parameters
    ----------
    array : T

    Returns
    -------
    Tuple[int, ...]
    """
    if isinstance(array, VECTOR_TYPES):
        return _shape_vec(array)
    elif isinstance(array, MATRIX_TYPES):
        return _shape_mat(array)
    elif isinstance(array, NDARRAY_TYPES):
        return array.shape
    else:
        raise TypeError(f"Unknown shape for array of type {type(array)}")

@require_fenics
@require_petsc
def _shape_vec(vec: Union[DfnVec, PETScVec]) -> Tuple[int]:
    """
    Return the vector shape

    This is just a tuple version of the `size_vec`

    Parameters
    ----------
    vec : dolfin.PETScVector, PETScMat or np.ndarray

    Returns
    -------
    Tuple[int]
    """
    return (_size_vec(vec),)

@require_fenics
@require_petsc
def _shape_mat(mat: Union[DfnMat, PETScMat]) -> Tuple[int, int]:
    """
    Return the matrix shape

    Parameters
    ----------
    generic_mat : PETScMat or np.ndarray

    Returns
    -------
    Tuple[int, int]
    """
    if isinstance(mat, PETScMat):
        return mat.getSize()
    elif isinstance(mat, DfnMat):
        return tuple([mat.size(i) for i in range(2)])
    else:
        raise TypeError(f"Unknown shape for matrix of type {type(mat)}")

## Specialized matrix/vector operations

def set_vec(veca: V, vecb: Union[DfnVec, PETScVec, np.ndarray, JaxArray]):
    """
    Set the specified values to a vector

    This handles the different indexing notations that vectors can have based on the type, shape,
    etc.

    Parameters
    ----------
    veca : V
    vecb : dolfin.PETScVector, PETScVec, np.ndarray
    """
    if isinstance(veca, NDARRAY_TYPES):
        # TODO: Raise a value error?
        # if len(veca.shape) != 1:
        #     raise ValueError(f"`veca` with shape {veca.shape} is not a vector")
        veca[:] = vecb
    elif isinstance(veca, PETScVec):
        veca.array[:] = vecb
    elif isinstance(veca, DfnVec):
        veca[:] = vecb
    else:
        raise TypeError(f"Unknown vector of type {type(veca)}")

# no `set_mat` is provided as setting matrix values is very specialized for
# sparse matrices

@require_petsc
def solve_petsc_lu(
        mat: PETScMat,
        b: PETScVec,
        out: Optional[PETScVec]=None,
        ksp: Optional['PETSc.KSP']=None
    ) -> Tuple[PETScVec, 'PETSc.KSP']:
    """
    Solve Ax=b using PETSc's LU solver
    """
    if ksp is None:
        ksp = PETSc.KSP().create()
        ksp.setType(ksp.Type.PREONLY)
        ksp.setOperators(mat)
        ksp.setUp()

        pc = ksp.getPC()
        pc.setType(pc.Type.LU)

    if out is None:
        out = mat.getVecRight()
    ksp.solve(b, out)
    return out, ksp

def mult_mat_vec(mat: M, vec: V, out: Optional[V]=None) -> V:
    """
    Return a matrix-vector product

    Parameters
    ----------
    mat : Union[dolfin.PETScMatrix, PETScMat, np.ndarray]
    vec : Union[dolfin.PETScVector, PETScVec, np.ndarray]

    Returns
    -------
    out : Union[dolfin.PETScVector, PETScVec, np.ndarray]
    """
    if isinstance(mat, NDARRAY_TYPES) and isinstance(vec, NDARRAY_TYPES):
        if out is None:
            out = np.dot(mat, vec)
        else:
            np.dot(mat, vec, out=out)
    elif isinstance(mat, PETScMat) and isinstance(vec, PETScVec):
        out = mat.createVecLeft() if out is None else out
        mat.mult(vec, out)
    elif isinstance(mat, DfnMat) and isinstance(vec, DfnVec):
        out = mat*vec
    else:
        raise TypeError(f"Unknown matrix-vector product between types {type(mat)} and {type(vec)}")
    return out

def mult_mat_mat(mata: M, matb: M, out: Optional[M]=None) -> M:
    """
    Return a matrix-matrix product

    Parameters
    ----------
    mata, matb : Union[dolfin.PETScMatrix, PETScMat, np.ndarray]

    Returns
    -------
    out : Union[dolfin.PETScMatrix, PETScMat, np.ndarray]
    """
    if isinstance(mata, NDARRAY_TYPES) and isinstance(matb, NDARRAY_TYPES):
        if out is None:
            out = mata@matb
        else:
            np.matmul(mata, matb, out=out)
    elif isinstance(mata, PETScMat) and isinstance(matb, PETScMat):
        out = mata*matb
    elif isinstance(mata, PETScMat) and isinstance(matb, PETScMat):
        out = mata*matb
    else:
        raise TypeError(f"Unknown matrix-matrix product between types {type(mata)} and {type(matb)}")
    return out

def norm_vec(vec: V) -> float:
    """
    Return the 2-norm of a vector

    Parameters
    ----------
    vec : Union[dolfin.PETScVector, PETScVec, np.ndarray]

    Returns
    -------
    float
    """
    if isinstance(vec, PETScVec):
        return vec.norm()
    elif isinstance(vec, DfnVec):
        return vec.norm('l2')
    elif isinstance(vec, NDARRAY_TYPES):
        return np.linalg.norm(vec)
    else:
        raise TypeError(f"Unknown norm for vector type {type(vec)}")

def norm_mat(mat: M) -> float:
    """
    Return the frobenius norm of a matrix

    Parameters
    ----------
    vec : Union[dolfin.PETScMatrix, PETScMat, np.ndarray]

    Returns
    -------
    float
    """
    if isinstance(mat, PETScMat):
        return mat.norm(norm_type=PETSc.NormType.FROBENIUS)
    elif isinstance(mat, DfnMat):
        return mat.mat().norm(norm_type=PETSc.NormType.FROBENIUS)
    elif isinstance(mat, NDARRAY_TYPES):
        return np.linalg.norm(mat)
    else:
        raise TypeError(f"Unknown norm for matrix type {type(mat)}")

## Convert matrix/vector types
@require_petsc
def convert_mat_to_petsc(mat: M, comm=None, keep_diagonal: bool=True) -> PETScMat:
    """
    Return a `PETScMat` representation of `mat`
    """
    if isinstance(mat, PETScMat):
        out = mat
    elif isinstance(mat, DfnMat):
        out = mat.mat()
    elif isinstance(mat, NDARRAY_TYPES):
        out = _numpy_mat_to_petsc_mat_via_csr(mat, comm=comm, keep_diagonal=keep_diagonal)
    else:
        raise TypeError(f"Can't convert matrix of type {type(mat)} to PETScMat")

    return out

@require_petsc
def convert_vec_to_petsc(vec: V, comm=None) -> PETScVec:
    """
    Return a `PETScVec` representation of `vec`
    """
    n = size(vec)
    if isinstance(vec, PETScVec):
        out = vec
    elif isinstance(vec, DfnVec):
        out = vec.vec()
    elif isinstance(vec, NDARRAY_TYPES):
        out = PETScVec().createSeq(n, comm=comm)
        out.setUp()
        out.array[:] = vec
        out.assemble()
    else:
        raise TypeError(f"Can't convert vector of type {type(vec)} to PETScVec")

    return out

@require_petsc
def _numpy_mat_to_petsc_mat_via_csr(mat: np.ndarray, comm=None, keep_diagonal: bool=True):
    # converting mat to a numpy array seems to signifcantly affect speed
    mat = np.array(mat)
    mat_shape = shape(mat)

    # Build the CSR format of the resulting matrix by adding only non-zero values
    COL_IDXS = np.arange(mat_shape[1], dtype=np.int32)
    nz_row_idxs = [current_row != 0 for current_row in mat]
    Js = [COL_IDXS[nz_row_idx] for nz_row_idx in nz_row_idxs]
    Vs = [current_row[nz_row_idx] for nz_row_idx, current_row in zip(nz_row_idxs, mat)]

    # number of nonzeros in each row
    nnz = [len(sub_v) for sub_v in Vs]
    Is = [0] + [ii for ii in np.cumsum(nnz)]

    I = np.array(Is, dtype=np.int32)
    J = np.concatenate(Js, dtype=np.int32)
    V = np.concatenate(Vs)

    out = PETScMat().createAIJ(mat_shape, comm=comm, csr=(I, J, V))
    out.assemble()
    return out

@require_petsc
def _numpy_mat_to_petsc_mat_via_setvalues(mat: np.ndarray, comm=None, keep_diagonal=True):
    # Converting mat to a numpy array seems to signifcantly affect speed
    mat = np.array(mat)
    mat_shape = shape(mat)
    is_square = mat_shape[0] == mat_shape[1]

    COL_IDXS = np.arange(mat_shape[1], dtype=np.int32)
    out = PETScMat().createAIJ(mat_shape, comm=comm)
    out.setUp()
    for ii in range(mat_shape[0]):
        current_row = mat[ii, :]
        idx_nonzero = current_row != 0

        rows = [ii]
        cols = COL_IDXS[idx_nonzero]
        vals = current_row[idx_nonzero]
        out.setValues(rows, cols, vals, addv=PETSc.InsertMode.ADD_VALUES)

    if keep_diagonal and is_square:
        for ii in range(mat_shape[0]):
            out.setValue(ii, ii, 0.0, addv=PETSc.InsertMode.ADD_VALUES)
    out.assemble()
    return out

## Convert vectors to row/column PETScMat
@require_petsc
def convert_vec_to_rowmat(vec: Union[PETScVec, np.ndarray], comm=None) -> PETScMat:
    """
    Return a row `PETScMat` representation of `vec`
    """
    if isinstance(vec, DfnVec):
        vec = np.array(vec.vec().array)
    elif isinstance(vec, PETScVec):
        vec = np.array(vec.array)
    elif isinstance(vec, NDARRAY_TYPES):
        pass
    else:
        raise TypeError(f"Unknown convertion for vector type {type(vec)}")

    return convert_mat_to_petsc(vec.reshape(1, vec.size), comm)

@require_petsc
def convert_vec_to_colmat(vec, comm=None):
    """
    Return a column `PETScMat` representation of `vec`
    """
    if isinstance(vec, DfnVec):
        vec = np.array(vec.vec().array)
    elif isinstance(vec, PETScVec):
        vec = np.array(vec.array)
    elif isinstance(vec, NDARRAY_TYPES):
        pass
    else:
        raise TypeError(f"Unknown convertion for vector type {type(vec)}")

    return convert_mat_to_petsc(vec.reshape(vec.size, 1), comm)

## Specialized PETScMat routines

# Utilities for making specific types of matrices
@require_petsc
def zero_mat(n: int, m: int, comm=None) -> PETScMat:
    """
    Return a null matrix
    """
    mat = PETScMat().create(comm=comm)
    mat.setSizes([n, m])
    mat.setUp()
    mat.assemble()
    return mat

@require_petsc
def diag_mat(n: int, diag: float=1.0, comm=None) -> PETScMat:
    """
    Return a diagonal matrix
    """
    diag_vec = PETSc.Vec().create(comm=comm)
    diag_vec.setSizes(n)
    diag_vec.setUp()
    diag_vec.array[:] = diag
    diag_vec.assemble()

    mat = PETScMat().create(comm=comm)
    mat.setSizes([n, n])
    mat.setUp()
    mat.setDiagonal(diag_vec)
    mat.assemble()
    return mat

@require_petsc
def ident_mat(n, comm=None) -> PETScMat:
    """
    Return an identity matrix
    """
    return diag_mat(n, diag=1.0, comm=comm)
