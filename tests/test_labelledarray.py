"""
Test the functionality of the lablleledarray.py module
"""
import math
from itertools import accumulate, product
import string

import pytest 

from blockarray import labelledarray as la
from blockarray.labelledarray import LabelledArray, flatten_array

@pytest.fixture()
def setup_labelledarray():
    """
    Return a pre-defined `LabelledArray` and reference data 
    """
    l, m, n = 2, 3, 4
    shape = (l, m, n)
    labels = (('a', 'b'), ('a', 'b', 'c'), ('a', 'b', 'c', 'd'))

    strides = [
        stride for stride
        in accumulate(shape[-1:0:-1], lambda a, b: a*b, initial=1)]
    strides = tuple(strides)[::-1]

    elements = string.ascii_lowercase[:math.prod(shape)]

    return LabelledArray(elements, shape, labels), (elements, shape, labels, strides)


def _flat(midx, strides):
    return sum([idx*stride for idx, stride in zip(midx, strides)])

def test_shape(setup_labelledarray):
    """
    Test the LabelledArray has the correct shape
    """
    array, (_, shape, *_) = setup_labelledarray
    print(f"test has shape {array.shape} and vals {array.flat}")
    assert array.shape == shape

def test_single_index(setup_labelledarray):
    """Test that indexing a single element produces the correct result"""
    array, (ARRAY, *_, CSTRIDES) = setup_labelledarray
    # loop through each single index and check that the right element is selected
    all_axis_int_indices = [range(axis_size) for axis_size in array.shape]
    all_axis_str_indices = [axis_labels for axis_labels in array.labels]
    for mindex_int, mindex_str in zip(
        product(*all_axis_int_indices), product(*all_axis_str_indices)):

        assert array[mindex_int] == ARRAY[_flat(mindex_int, CSTRIDES)]
        assert array[mindex_str] == ARRAY[_flat(mindex_int, CSTRIDES)]

def test_array_index(setup_labelledarray):
    """Test that indexing a sub-array produces the correct result"""
    array, _ = setup_labelledarray
    assert array[:].shape == array.shape
    assert array[:].flat == array.flat

    assert array[...].shape == array.shape
    assert array[...].flat == array.flat

    assert array[0:1, 0:1, 0:1].shape == (1, 1, 1)
    # assert array[0:1, 0:1, 0:1].flat ==

    assert array[0:1, 0:1, 0:3].shape == (1, 1, 3)
    # assert array[0:1, 0:1, 0:3].lat ==


    axis_idxs = (0, slice(0, 1), slice(0, 1))
    assert array[axis_idxs].f_shape == (-1, 1, 1)
    assert array[axis_idxs].f_shape == (-1, 1, 1)

    print(f"array[:, :, 0] has shape {array[:, :, 0].shape} and vals {array[:, :, 0].flat}")
    print(f"array[:, :, 1:2] has shape {array[:, :, 1:2].shape} and vals {array[:, :, 1:2].flat}")
    print(f"array[:, :, 0:1] has shape {array[:, :, 0:1].shape} and vals {array[:, :, 0:1].flat}")
    print(f"array[:, :, :] has shape {array[:, :, :].shape} and vals {array[:, :, :].flat}")
    print(f"array[:] has shape {array[:].shape} and vals {array[:].flat}")

    print(flatten_array([[1, 2, 3], [4, 5, 6]]))


## Tests for indexing internals
def test_expand_multidx():
    """
    Test that multi-index with ellipses/missing axes are expanded correctly
    """
    multidx = (..., slice(None))
    assert la.expand_multidx(multidx, (1, 1, 1, 1)) == (slice(None),)*4

    multidx = (slice(None),)
    assert la.expand_multidx(multidx, (1, 1, 1, 1)) == (slice(None),)*4

# Tests for lists (and/or single) indexes along a single axis
def test_conv_gen_to_std_idx():
    """
    Test conversion of general to standard indices for a single axis
    """
    # Set the test case of a size 10 1-dimensional array
    N = 10
    LABEL_TO_IDX = {label: idx for idx, label in enumerate(string.ascii_lowercase[:N])}

    # In each case below, `std_idx` is the correct output standard index based 
    # on the known array size of 10
    gen_idx = ['a', 'b', 4, -5]
    std_idx =  [0, 1, 4, 10-5]
    assert la.conv_gen_to_std_idx(gen_idx, LABEL_TO_IDX, N) == std_idx

    gen_idx = slice(1, 10)
    std_idx =  list(range(1, 10))
    assert la.conv_gen_to_std_idx(gen_idx, LABEL_TO_IDX, N) == std_idx

    gen_idx = 5
    std_idx =  5
    assert la.conv_gen_to_std_idx(gen_idx, LABEL_TO_IDX, N) == std_idx

    gen_idx = 'a'
    std_idx =  0
    assert la.conv_gen_to_std_idx(gen_idx, LABEL_TO_IDX, N) == std_idx

def test_conv_list_to_std_idx():
    """
    Test conversion of a list of general indices to standard indices for a single axis
    """
    # Set the test case of a size 10 1-dimensional array
    N = 10
    LABEL_TO_IDX = {label: idx for idx, label in enumerate(string.ascii_lowercase[:N])}

    gen_idx = ['a', 'b', 4, -5]
    std_idx =  [0, 1, 4, 10-5]
    assert la.conv_list_to_std_idx(gen_idx, LABEL_TO_IDX, N) == std_idx

def test_conv_slice_to_std_idx():
    """
    Test conversion of a slice to standard indices for a single axis
    """
    N = 10

    for idx in [slice(2, 5), slice(2, 6, 2), slice(None)]:
        assert la.conv_slice_to_std_idx(idx, N) == list(range(N))[idx]

# Tests for single indexes along a single axis
def test_conv_label_to_std_idx():
    """
    Test conversion of a label index to an integer index
    """
    N = 10
    label_to_idx = {label: idx for idx, label in enumerate(string.ascii_lowercase[:N])}

    std_idx = N-1
    gen_idx = string.ascii_lowercase[std_idx]
    assert la.conv_label_to_std_idx(gen_idx, label_to_idx, N) == std_idx

def test_conv_neg_to_std_idx():
    """
    Test conversion of a negative index to an integer index
    """
    N = 10
    assert la.conv_neg_to_std_idx(5, N) == 5
    assert la.conv_neg_to_std_idx(-5, N) == 5

def test_conv_slice_start_to_idx():
    """
    Test conversion of a slice start index to an integer index
    """
    N = 10
    assert la.conv_slice_start_to_idx(None, N) == 0

    start = 5
    assert la.conv_slice_start_to_idx(start, N) == start

    start = -2
    assert la.conv_slice_start_to_idx(start, N) == N - 2

def test_conv_slice_stop_to_idx():
    """
    Test conversion of a slice stop index to an integer index
    """
    N = 10
    assert la.conv_slice_stop_to_idx(None, N) == N

    stop = 5
    assert la.conv_slice_stop_to_idx(stop, N) == stop

    stop = -2
    assert la.conv_slice_stop_to_idx(stop, N) == N - 2


if __name__ == '__main__':
    test_shape(setup_labelledarray())
    test_single_index(setup_labelledarray())
    test_array_index(setup_labelledarray())

    test_expand_multidx()

    test_conv_gen_to_std_idx()

    test_conv_list_to_std_idx()
    test_conv_slice_to_std_idx()

    test_conv_label_to_std_idx()
    test_conv_neg_to_std_idx()
    test_conv_slice_start_to_idx()
    test_conv_slice_stop_to_idx()
