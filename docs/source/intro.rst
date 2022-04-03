************
Introduction
************

This package provides the ``BlockTensor`` object that makes it easier to work with tensors that logically consist of multiple blocks or subtensors.


Indexing
========
Indexing block tensors works similarly to indexing in ``numpy`` with some additional indexing methods due to the labelled blocks. For a single axis there are four ways to index the subtensors. For a block tensor object ``tensor`` with 3 blocks represented by subtensors ``(a, b, c)`` and labels ``('a', 'b', 'c')`` these are:
    * single index by label
        * ``tensor['a']`` returns subtensor ``a``
    * single index by integer
        * ``tensor[0]`` returns subtensor ``a``
        * ``tensor[-1]`` returns subtensor ``c``
    * range of indices by slice
        * ``tensor[0:2]`` returns a ``BlockTensor`` with subtensors ``(a, b)``
        * ``tensor[:]`` returns the same ``BlockTensor`` as ``tensor``
    * range of indices by list of single indices
        * ``tensor[['a', 'b']]`` returns a ``BlockTensor`` with subtensors ``(a, b)``
        * ``tensor[[0, 1]]`` returns a ``BlockTensor`` with subtensors ``(a, b)``
        * ``tensor[[0, -1]]`` returns a ``BlockTensor`` with subtensors ``(a, c)``
