import timeit

import petsc4py.PETSc as PETSc
import numpy as np

from blockarray import blockmat as bmat


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

def benchmark_create_bmat():
    """Create a BlockMatrix"""
    bmat.BlockMatrix(MATS, labels=(('a', 'b'), ('a', 'b')))
    return None

if __name__ == '__main__':
    print(globals())

    print(timeit.timeit('benchmark_create_bmat()', globals=globals(), number=100))
