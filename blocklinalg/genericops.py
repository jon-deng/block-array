"""
This module contains generic functions that should work across the different
vector/matrix objects from PETSc, numpy, and FEniCS

This is needed for BlockVector and BlockMatrix to work with different
subelements from the different packages.
"""

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
        return mat@vec
    else:
        try:
            return mat*vec
        except:
            raise

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
        try:
            return mata*matb
        except:
            raise


def norm_mat(mat):
    if isinstance(mat, PETSc.Mat):
        return mat.norm(norm_type=PETSc.NormType.FROBENIUS)
    else:
        return np.sqrt(np.sum(mat**2))

def shape_mat(mat):
    """
    Return matrix shape for different matrix types

    Parameters
    ----------
    generic_mat : PETSc.Mat or np.ndarray
    """
    if isinstance(PETSc.Mat):
        return mat.getSize()
    else:
        assert len(mat.shape) == 2
        return mat.shape
