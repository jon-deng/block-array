"""
Generic functions for operating on different vector/matrix/array objects
from PETSc, numpy, and FEniCS.
"""

from typing import TypeVar, Union, Tuple, Optional, Generic
import math
import functools

import numpy as np
from . import _HAS_PETSC, _HAS_FENICS, _HAS_JAX, require_petsc, require_fenics

if _HAS_JAX:
    from jax import numpy as jnp
if _HAS_PETSC:
    from petsc4py import PETSc
if _HAS_FENICS:
    import dolfin as dfn

from .typing import Shape, DfnMat, DfnVec, PETScMat, PETScVec, JaxArray

# pylint: disable=no-member

NDARRAY_TYPES = (np.ndarray, np.generic)
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

ALL_TYPES = NDARRAY_TYPES + VECTOR_TYPES + MATRIX_TYPES
ALL_VECTOR_TYPES = NDARRAY_TYPES + VECTOR_TYPES
ALL_MATRIX_TYPES = NDARRAY_TYPES + MATRIX_TYPES
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


## Wrapper array objects
def vectorize_func(func):
    """
    Return a vectorized unwrap function over object arrays

    This function is needed because blocks are stored in numpy object arrays.
    When elements of an object array are themselves numpy arrays (i.e. for
    blocks) numpy tries to unpack inner arrays which is not the desired
    behaviour here.
    """

    def vfunc(input):
        if isinstance(input, np.ndarray):
            # Treat numpy object arrays as containers of subarrays
            if input.dtype == object:
                ret = np.empty(input.shape, dtype=object)
                _ret = ret.reshape(-1)
                # Doing this avoids numpy accessing `x.__array__` while assigning
                # as in `ret.reshape(-1)[:] = [func(x) for x in input.flat]`
                # This can save a reasonable amount of time
                for ii, x in enumerate(input.flat):
                    _ret[ii] = func(x)
                return ret
            # Treat numpy arrays (not object dtype) as a single subarray
            else:
                return func(input)
        # Recursively apply `func` to lists/tuples
        elif isinstance(input, list):
            return [vfunc(x) for x in input]
        elif isinstance(input, tuple):
            return tuple(vfunc(x) for x in input)
        # For all other subarray types, just apply `func` to them
        else:
            return func(input)

    return vfunc


@vectorize_func
def wrap(array):
    if isinstance(array, GenericSubarray):
        return array
    elif isinstance(array, NDARRAY_TYPES):
        return NumpyArrayLike(array)
    elif isinstance(array, PETScMat):
        return PETScMatrix(array)
    elif isinstance(array, PETScVec):
        return PETScVector(array)
    elif isinstance(array, DfnMat):
        return DfnMatrix(array)
    elif isinstance(array, DfnVec):
        return DfnVector(array)
    else:
        raise TypeError(
            f"Couldn't find wrapper array type for array of type {type(array)}"
        )


@vectorize_func
def unwrap(array):
    if isinstance(array, GenericSubarray):
        return array.data
    else:
        return array


T = TypeVar('T')


class GenericSubarray(Generic[T]):
    shape: Shape
    size: int
    ndim: int
    data: T

    def __init__(self, array: T):
        self._data = array

    def __array__(self, dtype=None):
        raise NotImplementedError(
            f"`__array__` interface not implemented for array wrapper type {type(self)}"
        )

    def __getitem__(self, key):
        raise NotImplementedError(
            f"Can't index values from array wrapper type {type(self)}"
        )

    def __setitem__(self, key, value):
        raise NotImplementedError(
            f"Can't set at index to array wrapper type {type(self)}"
        )

    def set(self, value):
        """
        Set the array to `value`
        """
        raise NotImplementedError(
            f"Can't set values to array wrapper type {type(self)}"
        )

    ## Methods that are usually well defined

    def __len__(self):
        return self.data.__len__()

    def copy(self) -> 'GenericSubarray[T]':
        """Return a copy"""
        return type(self)(self.data.copy())

    @property
    def data(self) -> T:
        return self._data


class PETScVector(GenericSubarray[PETScVec]):
    def __init__(self, array: PETScVec):
        super().__init__(array)
        assert isinstance(self.data, PETScVec)

    def __array__(self, dtype=None):
        return np.array(self.data.array, dtype=dtype)

    def __getitem__(self, key):
        return self.data.array[key]

    def __setitem__(self, key, value):
        self.data.array[key] = value

    @property
    def shape(self):
        return (self.size,)

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return 1

    def set(self, value):
        self.data.array[:] = value


class PETScMatrix(GenericSubarray[PETScMat]):
    def __init__(self, array: PETScMat):
        super().__init__(array)
        assert isinstance(self.data, PETScMat)

    def __array__(self, dtype=None):
        return np.array(self.data[:, :], dtype=dtype)

    # def __getitem__(self, key):
    #     return self.data[key]

    # def __setitem__(self, key, value):
    #     self.data[key] = value

    @property
    def shape(self):
        return self.data.getSize()

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def ndim(self):
        return 2


class DfnVector(GenericSubarray[DfnVec]):
    def __init__(self, array: DfnVec):
        super().__init__(array)
        assert isinstance(self.data, DfnVec)

        self._shape = (self.data.size(),)

    def __array__(self, dtype=None):
        return np.array(self.data[:], dtype=dtype)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def ndim(self):
        return 1

    def set(self, value):
        self.data[:] = value


class DfnMatrix(GenericSubarray[DfnMat]):
    def __init__(self, array: DfnMat):
        super().__init__(array)
        assert isinstance(self.data, DfnMat)

    def __array__(self, dtype=None):
        return np.array(self.data[:, :], dtype=dtype)

    # def __getitem__(self, key):
    #     return self.data[key]

    # def __setitem__(self, key, value):
    #     self.data[key] = value

    @property
    def shape(self):
        return tuple(self.data.size(ii) for ii in range(2))

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def ndim(self):
        return 2


V = TypeVar('V', *NDARRAY_TYPES)


class NumpyArrayLike(GenericSubarray[V]):

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __array__(self, dtype=None):
        return np.array(self.data, dtype=dtype)

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    def set(self, value):
        self.data[:] = value

@require_petsc
def solve_petsc_preonly(
    mat: PETScMat,
    b: PETScVec,
    out: Optional[PETScVec] = None,
    ksp: Optional['PETSc.KSP'] = None,
    pc_type: str = 'lu'
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
        pc.setFactorSolverType(pc_type)

    if out is None:
        out = mat.getVecRight()
    ksp.solve(b, out)
    return out, ksp


@require_petsc
def solve_petsc_preonly_lu(
    mat: PETScMat,
    b: PETScVec,
    out: Optional[PETScVec] = None,
    ksp: Optional['PETSc.KSP'] = None,
) -> Tuple[PETScVec, 'PETSc.KSP']:
    """
    Solve Ax=b using PETSc's LU solver
    """
    return solve_petsc_preonly(mat, b, out, ksp, pc_type='lu')


@require_petsc
def solve_petsc_preonly_superlu(
    mat: PETScMat,
    b: PETScVec,
    out: Optional[PETScVec] = None,
    ksp: Optional['PETSc.KSP'] = None,
) -> Tuple[PETScVec, 'PETSc.KSP']:
    """
    Solve Ax=b using PETSc's LU solver
    """
    return solve_petsc_preonly(mat, b, out, ksp, pc_type='superlu')


def mult_mat_vec(mat: M, vec: V, out: Optional[V] = None) -> V:
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
        out = mat * vec
    else:
        raise TypeError(
            f"Unknown matrix-vector product between types {type(mat)} and {type(vec)}"
        )
    return out


def mult_mat_mat(mata: M, matb: M, out: Optional[M] = None) -> M:
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
            out = mata @ matb
        else:
            np.matmul(mata, matb, out=out)
    elif isinstance(mata, PETScMat) and isinstance(matb, PETScMat):
        out = mata * matb
    elif isinstance(mata, PETScMat) and isinstance(matb, PETScMat):
        out = mata * matb
    else:
        raise TypeError(
            f"Unknown matrix-matrix product between types {type(mata)} and {type(matb)}"
        )
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
def convert_mat_to_petsc(mat: M, comm=None, keep_diagonal: bool = True) -> PETScMat:
    """
    Return a `PETScMat` representation of `mat`
    """
    if isinstance(mat, PETScMat):
        out = mat
    elif isinstance(mat, DfnMat):
        out = mat.mat()
    elif isinstance(mat, NDARRAY_TYPES):
        out = _numpy_mat_to_petsc_mat_via_csr(
            mat, comm=comm, keep_diagonal=keep_diagonal
        )
    else:
        raise TypeError(f"Can't convert matrix of type {type(mat)} to PETScMat")

    return out


@require_petsc
def convert_vec_to_petsc(vec: V, comm=None) -> PETScVec:
    """
    Return a `PETScVec` representation of `vec`
    """
    if not isinstance(vec, GenericSubarray):
        vec = wrap(vec)

    n = vec.size
    vec = vec.data
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


def convert_vec_to_numpy(vec: V) -> np.ndarray:
    """
    Return a `np.ndarray` representation of `vec`
    """
    if not isinstance(vec, GenericSubarray):
        vec = wrap(vec)

    n = vec.size
    vec = vec.data
    if isinstance(vec, PETScVec):
        out = np.array(vec)
    elif isinstance(vec, DfnVec):
        out = np.array(vec)
    elif isinstance(vec, NDARRAY_TYPES):
        out = np.array(vec)
    else:
        raise TypeError(f"Can't convert vector of type {type(vec)} to numpy")

    return out


@require_petsc
def _numpy_mat_to_petsc_mat_via_csr(
    mat: np.ndarray, comm=None, keep_diagonal: bool = True
):
    """
    Return a `PETSc.Mat` from a 2D `numpy` array

    This also removes all non-zeros from the mat
    """
    # Converting `mat` to a numpy array first can signifcantly affect speed
    mat = np.array(mat)
    mat_shape = mat.shape

    # Build the CSR format of the resulting matrix by adding only non-zero values
    # from each row
    # This involves getting the `I`, `J`, `V` values for the sparse matrix
    # (the COO format, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html)

    # To find the column indices for each row:
    # create a boolean array indicating non-zeros in each row
    COL_IDXS = np.arange(mat_shape[1], dtype=np.int32)
    rows_is_nonzero = [current_row != 0 for current_row in mat]
    rows_j = [np.array(COL_IDXS[is_nonzero]) for is_nonzero in rows_is_nonzero]
    J = np.concatenate(rows_j, dtype=np.int32)

    # To find the values for each row:
    # use the boolean array indicating non-zeros in each row, to pick out the row values
    rows_v = [
        np.array(current_row[is_nonzero])
        for is_nonzero, current_row in zip(rows_is_nonzero, mat)
    ]
    V = np.concatenate(rows_v)

    # To find the row indices and the range they occupy in `I` and `J`:
    # Find the number of non-zeros in each row, `nnz`,
    # and use this to find the range of indices for each row
    nnz = [len(sub_v) for sub_v in rows_v]
    I = np.concatenate((np.array([0]), np.cumsum(nnz)), dtype=np.int32)

    out = PETScMat().createAIJ(mat_shape, comm=comm, csr=(I, J, V))
    out.assemble()
    return out


@require_petsc
def _numpy_mat_to_petsc_mat_via_setvalues(
    mat: np.ndarray, comm=None, keep_diagonal=True
):
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
    mat = PETScMat().createAIJ((n, m), nnz=0, comm=comm)
    mat.assemble()
    return mat


@require_petsc
def diag_mat(n: int, diag: float = 1.0, comm=None) -> PETScMat:
    """
    Return a diagonal matrix
    """
    diag_vec = PETSc.Vec().create(comm=comm)
    diag_vec.setSizes(n)
    # Calling `.setUp()` prevents a segmentation fault; I don't know enough
    # about PETSc to know why/when `setUp()` must be called for vectors though
    diag_vec.setUp()
    diag_vec.set(diag)
    diag_vec.assemble()

    # See https://petsc.org/release/docs/manual/mat/#sec-matsparse
    # for a description of what `nnz` is
    mat = PETScMat().createAIJ((n, n), nnz=1, comm=comm)

    # Note that if `nz`/`nnz` is not specified, then `mat.setUp()` must be
    # called, otherwise memory errors (seg fault) are usually triggered
    # mat.setUp()
    mat.setDiagonal(diag_vec)
    mat.assemble()
    return mat


@require_petsc
def ident_mat(n, comm=None) -> PETScMat:
    """
    Return an identity matrix
    """
    return diag_mat(n, diag=1.0, comm=comm)
