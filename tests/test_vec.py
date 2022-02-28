import numpy as np
import petsc4py.PETSc as PETSc

from blocklinalg import mat as bmat
from blocklinalg import linalg as bla
from blocklinalg import vec as bvec

a = np.arange(5)
b = np.arange(3)
c = np.arange(4)
VEC1 = bvec.BlockVec((a, b, c), ('a', 'b', 'c'))
def test_vec_size_shape():
    print(VEC1.size)
    print(VEC1.shape)
    print(VEC1.bsize)
    print(VEC1.bshape)

if __name__ == '__main__':
    test_vec_size_shape()