<h1 align="center">
<img src="docs/source/logo/BlockArray.png" width="300">
</h1><br>

BlockArray is a package for working with tensors logically partitioned into blocks (or subtensors) in a nested format. For example, block matrices and block vectors can be created as:
```python
from blockarray.blockarray import BlockArray
from blockarray.linalg import mult_mat_vec

# model the block vector
# [x0, x1]
# where x0 = np.array([1, 2, 3])
# x1 = np.array([4, 5])
x = BlockArray([np.array([1, 2, 3]), np.array([4, 5])])

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
A = BlockArray([[A00, A01], [A10, A11]])

# Basic math operations
y = mult_mat_vec(2*A, x)
z = x+y
```

## Motivation and Similar Projects

There are similar projects that provide block matrix and block vector functionality such as the excellent FEniCS project and its associated projects (https://github.com/FEniCS, https://fenicsproject.org/, https://bitbucket.org/fenics-apps/cbc.block). The PETSc project also provides a nested matrix and vector format (https://petsc.org/release/#). This package provides a more generic block array to facilititate higher dimensional arrays.
