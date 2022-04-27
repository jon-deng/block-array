************
Introduction
************

This package provides a ``BlockArray`` object that makes it easier to work with tensors that are logically divided into multiple blocks or subtensors.
The basic usage of the ``BlockArray`` is illustrated below for the case of a block matrix and block vector::

    import numpy as np
    from blockarray.tensor import BlockArray

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
    A = BlockArray([[A00, A01], [A10, A11]], labels=(('a', 'b'), ('a', 'b')))


    # `X` is a block representation of the vector
    # [1, 2, 3]
    # the first block is labelled 'a' and the second 'b'
    X0 = np.array([1, 2])
    X1 = np.array([3])
    X = BlockArray([X0, X1], labels=(('a', 'b'),))

In the above example, the matrix ``A`` is represented by 2 row blocks (with corresponding submatrix row sizes 2 and 1) and 2 column blocks (with corresponding submatrix column sizes 2 and 1) while the vector ``X`` is represented with two row blocks (with corresponding subvector sizes 2 and 1).
Blocks of ``A`` and ``X`` are also labelled; labels are useful to organize blocks since blocks often represent different different systems.
For example, the block 0 of ``X``, ``X0``, might correspond to a vector associate with system ``'a'`` while block 1 of ``X``, ``X1``, corresponds to a vector associated with system ``'b'``.

Indexing
========
Indexing block tensors works similarly to indexing in ``numpy`` with some additional indexing methods due to the labelled blocks. For a single axis there are four ways to index the subtensors. For a block tensor object ``tensor`` with 3 blocks represented by subtensors ``(a, b, c)`` and labels ``('a', 'b', 'c')`` these are:
    * single index by label
        * ``tensor['a']`` returns subtensor ``a``
    * single index by integer
        * ``tensor[0]`` returns subtensor ``a``
        * ``tensor[-1]`` returns subtensor ``c``
    * range of indices by slice
        * ``tensor[0:2]`` returns a ``BlockArray`` with subtensors ``(a, b)``
        * ``tensor[:]`` returns the same ``BlockArray`` as ``tensor``
    * range of indices by list of single indices
        * ``tensor[['a', 'b']]`` returns a ``BlockArray`` with subtensors ``(a, b)``
        * ``tensor[[0, 1]]`` returns a ``BlockArray`` with subtensors ``(a, b)``
        * ``tensor[[0, -1]]`` returns a ``BlockArray`` with subtensors ``(a, c)``
