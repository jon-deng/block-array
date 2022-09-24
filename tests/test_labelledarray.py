"""
Test the functionality of the lablleledarray.py module
"""
import math
from typing import TypeVar
from itertools import accumulate, product
import operator
import string

import pytest
import numpy as np

from blockarray import labelledarray as la
from blockarray.labelledarray import LabelledArray, expand_multi_gen_idx, conv_multi_gen_to_std_idx, flatten_array
from blockarray.typing import FlatArray, MultiLabelToStdIndex, Shape, MultiGenIndex

def flat_idx(midx, strides):
    """
    Return a flat index from a multi-index and strides
    """
    return np.sum(np.multiply(midx, strides))

def squeeze_shape(f_shape: Shape) -> Shape:
    """
    Return a shape without reduced axes from a full shape
    """
    return tuple(ax_size for ax_size in f_shape if ax_size != -1)

def strides_from_shape(shape: Shape) -> Shape:
    """
    Return a c-strides tuple from a shape
    """
    return tuple(accumulate(shape[1:][::-1], operator.mul, initial=1))[::-1]

T = TypeVar('T')
class TestLabelledArray:
    @pytest.fixture()
    def setup_labelledarray(self):
        """
        Return a `LabelledArray` instance and reference data for testing

        Returns
        -------
        LabelledArray
            The labelled array instance to test
        Tuple[elements, shape, labels, strides]
            A tuple of reference quantities used in constructing the `LabelledArray`
            instance. These should be used to test the array against.
        """
        l, m, n = 2, 3, 4
        shape = (l, m, n)
        labels = (('a', 'b'), ('a', 'b', 'c'), ('a', 'b', 'c', 'd'))

        strides = [
            stride for stride
            in accumulate(shape[-1:0:-1], lambda a, b: a*b, initial=1)]
        strides = tuple(strides[::-1])

        elements = string.ascii_lowercase[:math.prod(shape)]
        elements = [char for char in elements]

        return LabelledArray(elements, shape, labels), (elements, shape, labels, strides)

    def test_shape(self, setup_labelledarray):
        """
        Test the `LabelledArray` instance has the correct shape
        """
        array, (_, ref_f_shape, *_) = setup_labelledarray
        # print(f"test has shape {array.shape} and vals {array.array}")
        assert array.shape == squeeze_shape(ref_f_shape)

    def test_f_shape(self, setup_labelledarray):
        """
        Test the `LabelledArray` instance has the correct shape
        """
        array, (_, ref_f_shape, *_) = setup_labelledarray
        # print(f"test has shape {array.shape} and vals {array.array}")
        assert array.f_shape == ref_f_shape

    def test_ndim(self, setup_labelledarray):
        """
        Test the `LabelledArray` instance has the correct shape
        """
        array, (_, ref_f_shape, *_) = setup_labelledarray
        assert array.ndim == len(ref_f_shape)

    def test_f_ndim(self, setup_labelledarray):
        """
        Test the `LabelledArray` instance has the correct shape
        """
        array, (_, ref_f_shape, *_) = setup_labelledarray
        assert array.f_ndim == len(squeeze_shape(ref_f_shape))

    def test_single_elem_index(self, setup_labelledarray):
        """
        Test indexing a single element from a `LabelledArray`

        Note that the test for label indices won't work if the array instance
        doesn't have labels.
        """
        array, (ref_data, *_, ref_strides) = setup_labelledarray

        # Loop through each single index and check that the right element is selected
        # TODO: This won't work for arrays with axis labels
        all_axis_int_indices = [range(axis_size) for axis_size in array.shape]
        all_axis_str_indices = [axis_labels for axis_labels in array.labels]
        for mindex_int, mindex_str in zip(
                product(*all_axis_int_indices), product(*all_axis_str_indices)
            ):

            assert array[mindex_int] == ref_data[flat_idx(mindex_int, ref_strides)]
            assert array[mindex_str] == ref_data[flat_idx(mindex_int, ref_strides)]

    def test_multi_elem_index(self, setup_labelledarray):
        """
        Test multiple elements from a `LabelledArray`
        """
        array, _ = setup_labelledarray
        assert array[:].shape == array.shape
        assert np.all(array[:].array == array.array)

        assert array[...].shape == array.shape
        assert np.all(array[...] == array.array)

        assert array[0:1, 0:1, 0:1].shape == (1, 1, 1)
        assert array[0:1, 0:1, 0:3].shape == (1, 1, 3)

        axis_idxs = (0, slice(0, 1), slice(0, 1))
        assert array[axis_idxs].f_shape == (-1, 1, 1)

        axis_idxs = (0, slice(0, 1), slice(1, 2))
        assert array[axis_idxs].f_shape == (-1, 1, 1)

        print(f"array[:, :, 0] has shape {array[:, :, 0].shape} and vals {array[:, :, 0].array}")
        print(f"array[:, :, 1:2] has shape {array[:, :, 1:2].shape} and vals {array[:, :, 1:2].array}")
        print(f"array[:, :, 0:1] has shape {array[:, :, 0:1].shape} and vals {array[:, :, 0:1].array}")
        print(f"array[:, :, :] has shape {array[:, :, :].shape} and vals {array[:, :, :].array}")
        print(f"array[:] has shape {array[:].shape} and vals {array[:].array}")

        print(flatten_array([[1, 2, 3], [4, 5, 6]]))

    def test_index_generic(self, setup_labelledarray):
        array, (elements, shape, labels, strides) = setup_labelledarray

        mlabel_to_idx = array._MULTI_LABEL_TO_IDX
        test_idx = (0, 0)
        self._test_index(array, elements, test_idx, shape, mlabel_to_idx)


    @staticmethod
    def _test_index(
            array: LabelledArray[T],
            elements: FlatArray[T],
            midx: MultiGenIndex,
            shape: Shape,
            mlabel_to_idx: MultiLabelToStdIndex
        ):
        """
        Test a generic indexing method
        """
        strides = np.cumprod((1,)+shape[:-1][::-1])[::-1]

        ## Compute the reference result of the index
        # Compute a flat reference index to get the 'correct' elements
        def require_list(x):
            if isinstance(x, (list, tuple)):
                return x
            else:
                return [x]

        midx = expand_multi_gen_idx(midx, shape)
        ref_multi_std_idx = conv_multi_gen_to_std_idx(midx, shape, mlabel_to_idx)
        _ref_multi_std_idx = [require_list(x) for x in ref_multi_std_idx]

        def to_flat(midx):
            return np.sum(midx*strides)

        ref_idx_elements = [
            elements[to_flat(midx)] for midx in product(*_ref_multi_std_idx)
        ]

        ## Compute the `LabelledArray` index result
        ref_idx_elements = array[midx].array.tolist()

        ## Compare the reference and `LabelledArray` results to check
        assert ref_idx_elements == ref_idx_elements

# TODO: Add tests for `validate_*` functions

def test_flatten_array():
    """
    Test the `flatten_array` function
    """
    ref_array = [[1, 2, 3], [4, 5, 6]]
    ref_shape = (2, 3)
    ref_flat_array = [1, 2, 3, 4, 5, 6]
    flat_array, shape = la.flatten_array(ref_array)
    assert flat_array == ref_flat_array and shape == ref_shape

    ref_array = [[1, 2, 3, 4, 5, 6]]
    ref_shape = (1, 6)
    ref_flat_array = [1, 2, 3, 4, 5, 6]
    flat_array, shape = la.flatten_array(ref_array)
    assert flat_array == ref_flat_array and shape == ref_shape

def test_nest_array():
    """
    Test the `nest_array` function
    """
    ref_array = [[1, 2, 3], [4, 5, 6]]
    ref_shape = (2, 3)
    ref_strides = strides_from_shape(ref_shape)
    ref_flat_array = [1, 2, 3, 4, 5, 6]
    array = la.nest_array(ref_flat_array, ref_strides)
    assert array == ref_array

    ref_array = [[1, 2, 3, 4, 5, 6]]
    ref_shape = (1, 6)
    ref_strides = strides_from_shape(ref_shape)
    ref_flat_array = [1, 2, 3, 4, 5, 6]
    array = la.nest_array(ref_flat_array, ref_strides)
    assert array == ref_array

## Tests for indexing internals
def test_expand_multidx():
    """
    Test expansion of a multi-index with ellipses and/or missing axes
    """
    multidx = (..., slice(None))
    assert la.expand_multi_gen_idx(multidx, (1, 1, 1, 1)) == (slice(None),)*4

    multidx = (slice(None),)
    assert la.expand_multi_gen_idx(multidx, (1, 1, 1, 1)) == (slice(None),)*4

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
    Test conversion of a list index to standard indices for a single axis
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
