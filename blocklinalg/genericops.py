"""
This module contains generic functions that should work across the different
vector/matrix objects from PETSc, numpy, and FEniCS

This is needed for BlockVector and BlockMatrix to work with different
subelements from the different packages.
"""

def set_vec(veca, vecb):
    """
    Set the specified values to a vector

    This handles the different indexing notations that vectors can have based on the type, shape,
    etc.

    Parameters
    ----------
    vec : dolfin.PETScVector, PETsc.Vec, np.ndarray
    vals : float, array_like, etc.
    """
    if isinstance(veca, np.ndarray) and veca.shape == ():
        veca[()] = vecb
    else:
        veca[:] = vecb

def set_mat(A, B):
    """
    Set the specified values to a vector

    This handles the different indexing notations that vectors can have based on the type, shape,
    etc.

    Parameters
    ----------
    vec : dolfin.PETScVector, PETsc.Vec, np.ndarray
    vals : float, array_like, etc.
    """
    if isinstance(vec, np.ndarray) and vec.shape == ():
        vec[()] = vals
    else:
        vec[:] = vals

def add_vec(a, b, out=None):

    return out