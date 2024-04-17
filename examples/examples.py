"""
A collection of examples illustrating some use cases for blockarray
"""

import numpy as np

from blockarray import blockmat as bmat, blockvec as bvec
from blockarray import linalg as bla

## Storing block matrices
A00 = np.array([[1, 2], [4, 5]])
A01 = np.array([[3], [6]])
A10 = np.array([[7, 8], [3, 4]])
A11 = np.array([[9]])

A = bmat.BlockMatrix([[A00, A01], [A10, A11]], labels=(('a', 'b'), ('a', 'b')))

B = A.copy()

x0 = np.array([1, 2])
x1 = np.array([3])
x = bvec.BlockVector([x0, x1], labels=(('a', 'b'),))

## Basic math operation on matrices
C = A + B
C = 2 * A

## Basic tensor operations
y = bla.mult_mat_vec(C, x)
