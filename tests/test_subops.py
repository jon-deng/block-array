"""
Test `blockarray.subops`
"""

import pytest

from petsc4py import PETSc

from blockarray import subops


@pytest.fixture(
    params=[
        10,
    ]
)
def setup_mat(request):
    n = request.param
    mat = PETSc.Mat().createAIJ((n, n), nnz=3)
    for ii in range(1, n - 1):
        mat.setValues([ii - 1, ii, ii + 1], [ii], [0.5, 1, 0.5])

    mat.setValue(0, 0, 1)
    mat.setValue(n - 1, n - 1, 1)
    mat.assemble()
    return mat


def test_solve_petsc_lu_reuse_ksp(setup_mat):
    """
    Test `solve_petsc_lu` is consistent when reusing the `ksp` context
    """
    mat = setup_mat
    b = mat.getVecLeft()
    b.set(1.0)
    x1 = mat.getVecRight()
    x2 = mat.getVecRight()

    x1, ksp = subops.solve_petsc_preonly_lu(mat, b, x1)

    x2, ksp = subops.solve_petsc_preonly_lu(mat, b, x2, ksp=ksp)

    assert (x2 - x1).norm() == 0
