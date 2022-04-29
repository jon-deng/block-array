<h1 align="center">
<img src="docs/source/logo/blockarray_logo.svg" width="300">
</h1><br>

BlockArray is a package for working with arrays logically partitioned into blocks (or subarrays) in a nested format (also called block tensors or nested matrices, etc.). The main object is a `BlockArray` which can be created and indexed as shown:
```python
from blockarray.blockarray import BlockArray
from blockarray.linalg import mult_mat_vec

# model the block vector
# [x0, x1]
# where,
x0 = np.array([1, 2, 3])
x1 = np.array([4, 5])
x = BlockArray([x0, x1])

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

## Indexing
# Select the (0,) subarray. This is equal to x0
x[0] 

# Select the (0, 0) subarray. This is equal to A00
A[0, 0] 

# Select the (0, 0) subarray as a BlockArray with 1 subarray
A[0:1, 0:1] 

# Select the (0, 0) subarray as a BlockArray with 1 subarray
# Represents the block matrix
# [[A00]]
A[0:1, 0:1] 

# Select the upper row of the BlockArray (2 subarrays)
# Represents the block matrix
# [[A00, A01]]
A[0:1, :] 

# Select column zero of the BlockArray (2 subarrays)
# Represents the block matrix
# [[A00],
#  [A10]]
A[:, 0:1] 
```

In addition, basic math operations can be applied on `BlockArray`s. A numpy ufunc interface currently works for simple operations but is likely buggy and needs to be further tested.
```python
## Basic math operations
# Basic math operations (add, sub, scalar mul, etc) are defined
# A numpy ufunc interface also works for simple examples but needs to be tested 
# more thoroughly
y = mult_mat_vec(2*A, x)
z = x+y

y = np.matmul(A, x)
```

## Motivation and Similar Projects

There are similar projects that provide block matrix and block vector functionality such as the excellent FEniCS project and its associated projects (https://github.com/FEniCS, https://fenicsproject.org/, https://bitbucket.org/fenics-apps/cbc.block). The PETSc project also provides a nested matrix and vector format (https://petsc.org/release/#). This package provides a more generic block array to facilititate higher dimensional arrays.
