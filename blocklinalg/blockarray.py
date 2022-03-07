"""
A BlockArray is a multidimensional array of a fixed shape, (similar to numpy arrays), containing
arbitrary objects and with blocks indexed by keys
"""

from typing import TypeVar, Tuple
from itertools import product

import numpy as np
import math
import functools as ft

T = TypeVar("T")
Shape = Tuple[int, ...]

class BlockArray:
    """
    An N-dimensional array

    Parameters
    ----------
    array : tuple of objects, length N
    shape : tuple of ints, length N
    """

    def __init__(self, array: Tuple[T, ...], shape: Shape, labels: Tuple[Tuple[str, ...], ...]):
        # Validate the array shape and labels
        assert len(labels) == len(shape)
        for ii, axis_size in enumerate(shape):
            assert len(labels[ii]) == axis_size
        assert len(array) == math.prod(shape)

        # Assign basic data
        self._array = array
        self._shape = shape
        self._labels = labels

        # Compute convenience constants
        self._STRIDES = tuple([
            math.prod(self.shape[ii+1:], start=1) 
            for ii in range(len(self.shape))])
        self._MULTI_LABEL_TO_IDX = tuple([
            {key: ii for key, ii in zip(keys, idxs)} 
            for keys, idxs in zip(LABELS, [range(axis_size) for axis_size in SHAPE])])

    @property
    def array(self):
        return self._array

    @property
    def shape(self):
        return self._shape

    @property
    def labels(self):
        return self._labels

    @property 
    def size(self):
        return math.prod(self.shape)

    def __len__(self):
        return self.size

    def __getitem__(self, multi_idx):
        multi_idx = process_multi_idx(multi_idx, self.shape, self._MULTI_LABEL_TO_IDX)
        
        # Find the returned BlockArray's shape and labels
        ret_shape = tuple([len(axis_idxs) for axis_idxs in multi_idx if isinstance(axis_idxs, tuple)])
        ret_labels = tuple([
            tuple([self.labels[ii] for ii in axis_idxs])
            for axis_idxs in multi_idx if isinstance(axis_idxs, tuple)
        ])

        # enclose single ints in a list so it works with itertools
        multi_idx = [(idx,) if isinstance(idx, int) else idx for idx in multi_idx]
        ret_flat_idxs = [process_flat_idx(idx, self._STRIDES) for idx in product(*multi_idx)]

        ret_array = tuple([self.array[flat_idx] for flat_idx in ret_flat_idxs])
        return BlockArray(ret_array, ret_shape, ret_labels)


def process_flat_idx(multi_idx, strides):
    return sum([idx*stride for idx, stride in zip(multi_idx, strides)])

def process_multi_idx(multi_idx, shape, multi_label_to_idx):
    out_multi_idx = [
        process_axis_idx(index, axis_size, axis_label_to_idx) 
        for index, axis_size, axis_label_to_idx in zip(multi_idx, shape, multi_label_to_idx)]
    return tuple(out_multi_idx)

def process_axis_idx(idx, size, label_to_idx):
    """
    Converts one of the ways of indexing an axis to a numeric index
    """
    assert len(label_to_idx) == size

    if isinstance(idx, slice):
        return convert_slice(idx, size)
    elif isinstance(idx, (list, tuple)):
        return tuple([convert_label_idx(key, label_to_idx, size) for key in list])
    elif isinstance(idx, str):
        return convert_label_idx(key, label_to_idx, size)
    elif isinstance(idx, int):
        return convert_neg_idx(idx, size)
    else:
        raise TypeError(f"Unknown index {idx} of type {type(idx)}.")

def convert_slice(idx, size):
    start = convert_start_idx(idx.start, size)
    stop = convert_stop_idx(idx.stop, size)
    if idx.step is None:
        step = 1
    else:
        step = idx.step
    return tuple(range(start, stop, step))

def convert_label_idx(idx, label_to_idx, size):
    out_index = label_to_idx[idx]
    assert out_index >= 0 and out_index < size
    return out_index

def _convert_neg_idx(idx, size):
    if idx >= 0:
        out_idx = idx
    else:
        out_idx = size-idx
    return out_idx

def convert_neg_idx(idx, size):
    out_idx = _convert_neg_idx(idx, size)

    assert out_idx >= 0 and out_idx < size
    return out_idx

def convert_start_idx(idx, size):
    if idx is None:
        out_idx = 0
    else:
        out_idx = _convert_neg_idx(idx, size)
    return out_idx

def convert_stop_idx(idx, size):
    if idx is None:
        out_idx = size
    else:
        out_idx = _convert_neg_idx(idx, size)
    return out_idx


if __name__ == '__main__':
    l, m, n = 2, 3, 4
    SHAPE = (l, m, n)
    LABELS = (('a', 'b'), ('a', 'b', 'c'), ('a', 'b', 'c', 'd'))
    SIZE = math.prod(SHAPE)

    import string
    ARRAY = string.ascii_lowercase[:SIZE]

    test = BlockArray(ARRAY, SHAPE, LABELS)

    print(f"test has shape {test.shape} and vals {test.array}")
    print(f"test[:, :, 0] has shape {test[:, :, 0].shape} and vals {test[:, :, 0].array}")
    print(f"test[:, :, 1:2] has shape {test[:, :, 1:2].shape} and vals {test[:, :, 1:2].array}")
    print(f"test[:, :, 0:1] has shape {test[:, :, 0:1].shape} and vals {test[:, :, 0:1].array}")
    