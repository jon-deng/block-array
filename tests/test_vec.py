import numpy as np
import petsc4py.PETSc as PETSc

from blocklinalg import mat as bmat
from blocklinalg import linalg as bla
from blocklinalg import vec as bvec

a = np.arange(5)
b = np.arange(3)
c = np.arange(4)
VEC1 = bvec.BlockVec((a, b, c), ('a', 'b', 'c'))

a = np.arange(5)+1
b = np.arange(3)+2
c = np.arange(4)+3
VEC2 = bvec.BlockVec((a, b, c), ('a', 'b', 'c'))

def test_vec_size_shape():
    print(VEC1.size)
    print(VEC1.shape)
    print(VEC1.bsize)
    print(VEC1.bshape)

def test_vec_add():
    print(VEC1+VEC2)

def test_vec_set():
    VEC1['a'] = 5
    print(VEC1)

if __name__ == '__main__':
    test_vec_size_shape()
    test_vec_add()
    test_vec_set()