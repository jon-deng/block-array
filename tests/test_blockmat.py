"""
Test the operations in `blockarray.blockmat`
"""

import pytest

import numpy as np
import petsc4py.PETSc as PETSc

from blockarray import subops
from blockarray import blockmat as bmat
from blockarray import linalg as bla

def _setup_submats_petsc():
    """
    Return submatrices A, B, C, D of a block matrix
    """
    COMM = PETSc.COMM_WORLD
    A = PETSc.Mat().create(COMM)
    A.setSizes([3, 3])
    A.setType('aij')
    A.setUp()

    for nrow in range(A.getSize()[0]):
        rows = np.array([nrow], dtype=np.int32)
        cols = np.array([0, 1], dtype=np.int32)
        vals = np.array([1.2, 2.4])
        A.setValues(rows, cols, vals)
    A.assemble()

    B = PETSc.Mat().create(COMM)
    B.setSizes([3, 2])
    B.setType('aij')
    B.setUp()
    B.setValues([0], [0], [5])
    B.assemble()

    C = PETSc.Mat().create(COMM)
    C.setSizes([2, 3])
    C.setUp()
    C.setValues([0], [0], [5])
    C.assemble()

    D = PETSc.Mat().create(COMM)
    D.setSizes([2, 2])
    D.setUp()
    D.setValues([0], [0], [2.0])
    D.assemble()
    return A, B, C, D

@pytest.fixture()
def setup_mat_petsc():
    """
    Return a `BlockMatrix` instance with PETSc submatrices
    """
    submats = _setup_submats_petsc()
    return bmat.BlockMatrix(submats, (2, 2), labels=(('a', 'b'), ('a', 'b')))

@pytest.fixture()
def setup_mat_petsc_pair(setup_mat_petsc):
    """
    Return two `BlockMatrix` instances with PETSc submatrices
    """
    mata = setup_mat_petsc
    matb = mata.copy()
    return mata, matb

def test_add(setup_mat_petsc_pair):
    """
    Test addition of two `BlockMatrix` instances
    """
    mata, matb = setup_mat_petsc_pair
    _res = mata + matb
    print(f"A: {_res.to_mono_petsc()[:, :]}")
    print(f"B: {_res.to_mono_petsc()[:, :]}")
    print(f"A+B: {_res.to_mono_petsc()[:, :]}")

def test_concatenate_mat(setup_mat_petsc_pair):
    """
    Test concatenation of two `BlockMatrix` instances
    """
    mata, matb = setup_mat_petsc_pair
    cbmat = bmat.concatenate_mat([[mata], [matb]], labels=[['a', 'b', 'c', 'd'], ['a', 'b']])
    print(cbmat.f_shape)

def test_mult_mat(setup_mat_petsc_pair):
    """
    Test matrix-matrix multiplication of two `BlockMatrix` instances
    """
    mata, matb = setup_mat_petsc_pair
    out = bla.mult_mat_mat(mata, matb)
    print(out.f_shape)

def test_transpose(setup_mat_petsc):
    """
    Test `BlockMatrix.tranpose`
    """
    mat = setup_mat_petsc

    print(mat.f_shape)
    D = mat.transpose()
    print(D.to_mono_petsc()[:, :])
    print(mat.to_mono_petsc()[:, :])

def test_to_mono_petsc_aij(setup_mat_petsc):
    """
    Test converison of `BlockMatrix` to monolithic PETSc format
    """
    mat = setup_mat_petsc
    print(bmat.to_mono_petsc(mat))
