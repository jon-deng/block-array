"""
Test the functionality of the array.py module
"""

from ast import Slice
from itertools import accumulate, product
import string

from blockarray import labelledarray as la
from blockarray.labelledarray import LabelledArray, flatten_array, nest_array

import math

l, m, n = 2, 3, 4
SHAPE = (l, m, n)
LABELS = (('a', 'b'), ('a', 'b', 'c'), ('a', 'b', 'c', 'd'))
SIZE = math.prod(SHAPE)

CSTRIDES = [
    stride for stride
    in accumulate(SHAPE[-1:0:-1], lambda a, b: a*b, initial=1)]
CSTRIDES = tuple(CSTRIDES)[::-1]

ARRAY = string.ascii_lowercase[:SIZE]

array = LabelledArray(ARRAY, SHAPE, LABELS)

def _flat(midx, strides):
    return sum([idx*stride for idx, stride in zip(midx, strides)])

def test_shape():
    print(f"test has shape {array.shape} and vals {array.flat}")
    assert math.prod(array.shape) == SIZE

def test_single_index():
    """Test that indexing a single element produces the correct result"""
    # loop through each single index and check that the right element is selected
    all_axis_int_indices = [range(axis_size) for axis_size in array.shape]
    all_axis_str_indices = [axis_labels for axis_labels in LABELS]
    for mindex_int, mindex_str in zip(
        product(*all_axis_int_indices), product(*all_axis_str_indices)):

        assert array[mindex_int] == ARRAY[_flat(mindex_int, CSTRIDES)]
        assert array[mindex_str] == ARRAY[_flat(mindex_int, CSTRIDES)]

def test_array_index():
    """Test that indexing a sub-array produces the correct result"""
    assert array[:].shape == array.shape
    assert array[:].flat == array.flat

    assert array[...].shape == array.shape
    assert array[...].flat == array.flat

    assert array[0:1, 0:1, 0:1].shape == (1, 1, 1)
    # assert array[0:1, 0:1, 0:1].flat ==

    assert array[0:1, 0:1, 0:3].shape == (1, 1, 3)
    # assert array[0:1, 0:1, 0:3].lat ==


    axis_idxs = (0, slice(0, 1), slice(0, 1))
    assert array[axis_idxs].shape == (-1, 1, 1)
    assert array[axis_idxs].shape == (-1, 1, 1)

    print(f"array[:, :, 0] has shape {array[:, :, 0].shape} and vals {array[:, :, 0].flat}")
    print(f"array[:, :, 1:2] has shape {array[:, :, 1:2].shape} and vals {array[:, :, 1:2].flat}")
    print(f"array[:, :, 0:1] has shape {array[:, :, 0:1].shape} and vals {array[:, :, 0:1].flat}")
    print(f"array[:, :, :] has shape {array[:, :, :].shape} and vals {array[:, :, :].flat}")
    print(f"array[:] has shape {array[:].shape} and vals {array[:].flat}")

    print(flatten_array([[1, 2, 3], [4, 5, 6]]))


## Tests for indexing internals
def test_multi_to_flat_idx():
    # shape and strides input by inspection
    shape = (1, 2, 3, 4)
    strides = (24, 12, 4, 1)
    midx = (
        [0, 1, 2, 3],
        [4],
        [1, 2]
    )

    flat_idxs = _flat(midx, strides)
    assert la.multi_to_flat_idx(flat_idxs, strides) == flat_idxs

def test_expand_multidx():
    multidx = (..., slice(None))
    assert la.expand_multidx(multidx, (1, 1, 1, 1)) == (slice(None),)*4

    multidx = (slice(None),)
    assert la.expand_multidx(multidx, (1, 1, 1, 1)) == (slice(None),)*4

# Test for the converion of a single general index to a standard on
def test_conv_gen_to_std_idx():
    N = 10
    label_to_idx = {label: idx for idx, label in enumerate(string.ascii_lowercase[:N])}

    idx = ['a', 'b', 4, -5]
    _idx =  [0, 1, 4, 10-5]
    assert la.conv_gen_to_std_idx(idx, label_to_idx, N) == _idx

    idx = slice(1, 10)
    _idx =  list(range(1, 10))
    assert la.conv_gen_to_std_idx(idx, label_to_idx, N) == _idx

    idx = 5
    _idx =  5
    assert la.conv_gen_to_std_idx(idx, label_to_idx, N) == _idx

    idx = 'a'
    _idx =  0
    assert la.conv_gen_to_std_idx(idx, label_to_idx, N) == _idx

# Tests for converting a sequence of indices
def test_conv_list_to_std_idx():
    N = 10
    label_to_idx = {label: idx for idx, label in enumerate(string.ascii_lowercase[:N])}

    idx = ['a', 'b', 4, -5]
    _idx =  [0, 1, 4, 10-5]
    assert la.conv_list_to_std_idx(idx, label_to_idx, N) == _idx

def test_conv_slice_to_std_idx():
    N = 10

    for IDX in [slice(2, 5), slice(2, 6, 2), slice(None)]:
        assert la.conv_slice_to_std_idx(IDX, N) == list(range(N))[IDX]

# Tests for converting a single index
def test_conv_label_to_std_idx():
    N = 10
    label_to_idx = {label: idx for idx, label in enumerate(string.ascii_lowercase[:N])}
    n = N-1
    assert la.conv_label_to_std_idx(string.ascii_lowercase[n], label_to_idx, N) == n

def test_conv_neg_to_std_idx():
    N = 10
    assert la.conv_neg_to_std_idx(5, N) == 5
    assert la.conv_neg_to_std_idx(-5, N) == 5

def test_conv_slice_start_to_idx():
    N = 10
    assert la.conv_slice_start_to_idx(None, N) == 0

    start = 5
    assert la.conv_slice_start_to_idx(start, N) == start

    start = -2
    assert la.conv_slice_start_to_idx(start, N) == N - 2

def test_conv_slice_stop_to_idx():
    N = 10
    assert la.conv_slice_stop_to_idx(None, N) == N

    stop = 5
    assert la.conv_slice_stop_to_idx(stop, N) == stop

    stop = -2
    assert la.conv_slice_stop_to_idx(stop, N) == N - 2


if __name__ == '__main__':
    test_shape()
    test_single_index()
    test_array_index()

    test_expand_multidx()

    test_conv_gen_to_std_idx()

    test_conv_list_to_std_idx()
    test_conv_slice_to_std_idx()

    test_conv_label_to_std_idx()
    test_conv_neg_to_std_idx()
    test_conv_slice_start_to_idx()
    test_conv_slice_stop_to_idx()
