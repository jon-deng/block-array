"""
A BlockArray is a multidimensional array of a fixed shape, (similar to numpy arrays), containing
arbitrary objects and with blocks indexed by keys
"""

import numpy as np
import math
import functools as ft

l, m, n = 2, 3, 4
SHAPE = (l, m, n)
LABELS = (('a', 'b'), ('a', 'b', 'c'), ('a', 'b', 'c', 'd'))

SIZE = math.prod(SHAPE)
MULTI_LABEL_TO_IDX = tuple(
    [{key: ii for key, ii in zip(keys, idxs)} 
    for keys, idxs in zip(LABELS, [range(axis_size) for axis_size in SHAPE])])

import string
ARRAY = string.ascii_lowercase[:SIZE]

NDIM = len(SHAPE)
STRIDES = tuple([math.prod(SHAPE[ii+1:], start=1) for ii in range(len(SHAPE))])


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
    # print(STRIDES)
    multi_index = (0, 0, 3)
    multi_index = (slice(0, None), slice(0, 1), 3)
    multi_index = (1, slice(0, 2), 3)

    print(process_multi_idx(multi_index, SHAPE, MULTI_LABEL_TO_IDX))

    # print(test[process_flat_idx(multi_index, STRIDES)])