"""
Test the functionality of the array.py module
"""

from itertools import accumulate, product
import string

from blocktensor.array import LabelledArray, flatten_array, nest_array

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
    print(f"test has shape {array.shape} and vals {array.array_flat}")
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
    assert array[:].array_flat == array.array_flat

    assert array[...].shape == array.shape
    assert array[...].array_flat == array.array_flat

    assert array[0:1, 0:1, 0:1].shape == (1, 1, 1)
    # assert array[0:1, 0:1, 0:1].array ==

    assert array[0:1, 0:1, 0:3].shape == (1, 1, 3)
    # assert array[0:1, 0:1, 0:3].array ==


    axis_idxs = (0, slice(0, 1), slice(0, 1))
    assert array[axis_idxs].shape == (-1, 1, 1)
    assert array[axis_idxs].shape == (-1, 1, 1)

    print(f"array[:, :, 0] has shape {array[:, :, 0].shape} and vals {array[:, :, 0].array_flat}")
    print(f"array[:, :, 1:2] has shape {array[:, :, 1:2].shape} and vals {array[:, :, 1:2].array_flat}")
    print(f"array[:, :, 0:1] has shape {array[:, :, 0:1].shape} and vals {array[:, :, 0:1].array_flat}")
    print(f"array[:, :, :] has shape {array[:, :, :].shape} and vals {array[:, :, :].array_flat}")
    print(f"array[:] has shape {array[:].shape} and vals {array[:].array_flat}")

    print(flatten_array([[1, 2, 3], [4, 5, 6]]))

if __name__ == '__main__':
    test_shape()
    test_single_index()
    test_array_index()
