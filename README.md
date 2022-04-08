# BlockTensor

BlockTensor is a package for working with tensors logically partitioned into blocks (or subtensors) in a nested format. For example, block matrices and block vectors can be created as:
```python
from blocktensor.tensor import BlockTensor
from blocktensor.linalg import mult_mat_vec

# model the block vector
# [x0, x1]
# where x0 = np.array([1, 2, 3])
# x1 = np.array([4, 5])
x = BlockTensor([np.array([1, 2, 3]), np.array([4, 5])])

# model the block matrix
# [[A00, A01],
#  [A10, A11]]
# where
A00 = np.array(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]])
A01 = np.array(
    [[1, 2],
     [3, 4],
     [5, 6]])
A10 = np.array(
    [[1, 2, 3],
     [4, 5, 6]])
A11 = np.array(
    [[1, 2],
     [3, 4]])
A = BlockTensor([[A00, A01], [A10, A11]])

# Basic math operations
y = mult_mat_vec(2*A, x)
z = x+y
```

## Motivation and Similar Projects

There are similar projects that provide block matrix and block vector functionality such as the excellent FEniCS project and it's associated projects (https://github.com/FEniCS, https://fenicsproject.org/, https://bitbucket.org/fenics-apps/cbc.block). These are typically specialized to the solution of PDE systems and sparse matrix/vector formats so can be difficult to apply outside of that use case. This package provides a more generic block tensor compatible with numpy to facilitate these other use cases.
