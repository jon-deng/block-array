import numpy as np
import petsc4py.PETSc as PETSc

from blockarray import subops
from blockarray import blockmat as bmat
from blockarray import linalg as bla

COMM = PETSc.COMM_WORLD
A = PETSc.Mat().create(COMM)
A.setSizes([3, 3])
# A.setPreallocationNNZ(np.array([2, 1, 1], dtype=np.int32)[None, :])
o_nnz = np.array([2, 1, 1], dtype=np.int32)
d_nnz = np.zeros(3, dtype=np.int32)
# A.setPreallocationNNZ((o_nnz, d_nnz))
# A.setPreallocationNNZ([2, 5, 5])
# A.setPreallocationNNZ(1)
A.setType('aij')
A.setUp()
# print(A.nnz())
# print(A.getInfo())

print(A.getOwnershipRange())
for nrow in range(A.getSize()[0]):
    rows = np.array([nrow], dtype=np.int32)
    cols = np.array([0, 1], dtype=np.int32)
    vals = np.array([1.2, 2.4])
    A.setValues(rows, cols, vals)

A.assemble()
# A.assemblyBegin()
# A.assemblyEnd()
# print(A.getInfo())
print(A[:, :])

B = PETSc.Mat().create(COMM)
B.setSizes([3, 2])
B.setType('aij')
B.setUp()
B.setValues([0], [0], [5])
B.assemble()
print(B[:, :])

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

MATS = \
    [[A, B],
     [C, D]]

BMAT1 = bmat.BlockMatrix(MATS, labels=(('a', 'b'), ('a', 'b')))
BMAT2 = bmat.BlockMatrix(MATS, labels=(('a', 'b'), ('a', 'b')))
BMAT3 = BMAT1+BMAT2

print(BMAT1.to_mono_petsc()[:, :])
print(BMAT3.to_mono_petsc()[:, :])

def test_mat_size_shape():
    print(BMAT1.size)
    print(BMAT1.f_shape)
    print(BMAT1.mshape)
    print(BMAT1.f_bshape)

def test_add():
    BMAT3 = BMAT1 + BMAT2
    print(f"A: {BMAT1[:, :].to_mono_petsc()[:, :]}")
    print(f"B: {BMAT1[:, :].to_mono_petsc()[:, :]}")
    print(f"A+B: {BMAT3[:, :].to_mono_petsc()[:, :]}")

def test_zero_mat():
    print(subops.zero_mat(5, 6)[:, :])

def test_ident_mat():
    print(subops.ident_mat(5)[:, :])

def test_concatenate_mat():
    cbmat = bmat.concatenate_mat([[BMAT1], [BMAT2]], labels=[['a', 'b', 'c', 'd'], ['a', 'b']])
    print(cbmat.f_shape)
    print(BMAT1.f_shape)
    # print(cbmat.to_petsc())

def test_mult_mat():
    out = bla.mult_mat_mat(BMAT1, BMAT2)
    print(out.f_shape)

def test_transpose():

    print(BMAT1.f_shape)
    D = BMAT1.transpose()
    print(D.to_mono_petsc()[:, :])
    print(BMAT1.to_mono_petsc()[:, :])

def test_to_mono_petsc_aij():
    print(bmat.to_mono_petsc(BMAT1))
    # print(bmat.to_mono_petsc_aij(BMAT1))

if __name__ == '__main__':
    test_mat_size_shape()
    test_zero_mat()
    test_ident_mat()
    test_concatenate_mat()
    test_transpose()
    test_to_mono_petsc_aij()