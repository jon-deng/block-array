************
Introduction
************

This package provides a ``BlockArray`` object that makes it easier to work with tensors that are logically divided into multiple blocks or subarrays. To represent a block array, ``BlockArray`` acts as a container to store the underlying subarrays. To illustrate this, consider the creation of matrix and vector ``BlockArray`` objects as shown below::

    import numpy as np
    import blockarray.blockarray as ba

    # `A` is a block representation of the matrix
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    # in the row dimension, blocks are labelled 'a' and 'b'
    # similarly in the column dimension, blocks are also labelled 'a' and 'b'
    A00 = np.array(
        [[1, 2],
         [4, 5]])
    A01 = np.array(
        [[3],
         [6]])
    A10 = np.array(
        [[7, 8]])
    A11 = np.array(
        [[9]])
    A = ba.BlockArray([[A00, A01], [A10, A11]], labels=(('a', 'b'), ('a', 'b')))
    A = ba.BlockArray([A00, A01, A10, A11], shape=(2, 2), labels=(('a', 'b'), ('a', 'b')))
    A.shape == (2, 2)
    A.bshape == ((3, 2), (3, 2))

    # `X` is a block representation of the vector
    # [1, 2, 3]
    # the first block is labelled 'a' and the second 'b'
    X0 = np.array([1, 2])
    X1 = np.array([3])
    X = ba.BlockArray([X0, X1], labels=(('a', 'b'),))
    X = ba.BlockArray([X0, X1], shape=(2,), labels=(('a', 'b'),))
    X.shape == (2,)
    X.bshape == ((3, 2),)

In the above example, the matrix ``A`` is represented by 2 row blocks (with corresponding submatrix row sizes 2 and 1) and 2 column blocks (with corresponding submatrix column sizes 2 and 1) while the vector ``X`` is represented with two row blocks (with corresponding subvector sizes 2 and 1).
Blocks of ``A`` and ``X`` are also labelled; labels are useful to organize blocks since blocks often represent different systems.
For example, block 0 of ``X``, ``X0``, might correspond to a vector associated with system ``'a'`` while block 1 of ``X``, ``X1``, corresponds to a vector associated with system ``'b'``.

The ``shape`` and ``bshape`` attributes store the layout of the blocks. ``shape`` represents the number of blocks along each dimension. ``bshape`` gives the axis size of each block, along each axis.

Indexing
========

Indexing block arrays works similarly to indexing in ``numpy`` with some additional indexing methods due to the labelled indices. For a single dimensional ``BlockArray`` there are four ways to index the subarrays. For a ``BlockArray`` ``x`` with 3 blocks represented by subarrays ``(a, b, c)`` and labels ``('a', 'b', 'c')`` these are:

    * single index by label (string)
        * ``x['a']`` returns subarray ``a``
    * single index by integer
        * ``x[0]`` returns subarray ``a``
        * ``x[-1]`` returns subarray ``c``
    * range of indices by slice
        * ``x[0:2]`` returns a ``BlockArray`` with subarrays ``(a, b)``
        * ``x[:]`` returns the same ``BlockArray`` as ``x``
    * range of indices by list of single indices
        * ``x[['a', 'b']]`` returns a ``BlockArray`` with subarrays ``(a, b)``
        * ``x[[0, 1]]`` returns a ``BlockArray`` with subarrays ``(a, b)``
        * ``x[[0, -1]]`` returns a ``BlockArray`` with subarrays ``(a, c)``

Indexing a single subarray returns the appropriate subarray, while indexing a range of indices returns another ``BlockArray`` but with a (potentially) different shape.

Multidimensional indexing works in a similar fashion; one of the four indexing methods above can be used in each axis to extract a single subarray, or a ``BlockArray``. The range of subarrays selected is then the set of subarrays that lie in the indices for each axis. To demonstrate this, for a ``BlockArray`` ``x`` with shape ``(1, 2, 3, 4)`` and labels ``(('a',), ('a', 'b'), ('a', 'b', 'c'), ('a', 'b', 'c', 'd'))``:

    * single index for each axis by string/integer
        * ``x[0, 0, 0, 3]`` returns the (0, 0, 0, 3) subarray
        * ``x['a', 'a', 'a', 3]`` returns the (0, 0, 0, 3) subarray
        * ``x['a', 'a', 'a', 'd']`` returns the (0, 0, 0, 3) subarray
    * range of indices for each axis
        * ``x[0:1, 0:2, 0:3, 0:3]`` returns a ``BlockArray`` with ``shape`` ``(1, 2, 3, 3)``
        * ``x[['a'], ['a', 'b'], ['a', 'b', 'c'], 0:3]`` returns a ``BlockArray`` with ``shape`` ``(1, 2, 3, 3)``
        * ``x[['a'], ['a', 'b'], ['a', 'b', 'c'], ['a', 'b', 'c']]`` returns a ``BlockArray`` with ``shape`` ``(1, 2, 3, 3)``
    * range of indices for some axes and a single index for others
        * ``x[0:1, 0, 0:3, 0:3]`` returns a ``BlockArray`` with ``f_shape`` ``(1, -1, 3, 3)`` and ``shape`` ``(1, 3, 3)``
        * ``x[0:1, 0, 3, 0:3]`` returns a ``BlockArray`` with ``f_shape`` ``(1, -1, -1, 3)`` and ``shape`` ``(1, 3)``

A special case occurs in the last example when mixing a single index and range of indices.
In this case, the axis indexed with a single index results in an axis size of -1 and is collapsed/reduced. This is needed since the dimension of the blocks should match the dimension of the underlying subarrays. The ``BlockArray.shape`` attribute stores the shape of non-collapsed axes while ``BlockArray.f_shape`` includes collapsed axes indicated by a size -1. Indexing implictly occurs only over non-collapsed axes. For example, for a ``BlockArray`` ``x`` with 'full' shape ``(-1, 2, -1, 4)`` (consisting of a total of 8=2*4 subarrays), ``x[1, 3]`` selects the index 1 from axis 1, and index 3 from axis 3.

Lastly, missing axis indices are also automatically expanded, similar to ``numpy``. To illustrate this, consider a ``BlockArray`` ``x`` with ``shape`` ``(1, 2, 3, 4)``:

    * provide fewer indices than the number of dimensions
        * ``x[0, 0]`` is equivalent to ``x[0, 0, :, :]``
        * ``x[0, 0:1, :]`` is equivalent to ``x[0, 0:1, :, :]``
    * use a single ellipsis to expand dimensions
        * ``x[0, ..., 0]`` is equivalent to ``x[0, :, :, 0]``
        * ``x[..., 0, 0]`` is equivalent to ``x[:, :, 0, 0]``
