"""
Test functionality of `blockarray.labelledarray`
"""
import math
from typing import TypeVar
from itertools import accumulate, product
import operator
import string

import pytest
import numpy as np

from blockarray import labelledarray as la
from blockarray.labelledarray import (
    LabelledArray, 
    expand_multi_gen_idx, 
    conv_multi_gen_to_std_idx, 
    flatten_array
)
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
    """
    Test `LabelledArray` functionality
    """

    @pytest.fixture()
    def setup_array(self):
        """
        Return a `LabelledArray` instance and reference data for testing

        Returns
        -------
        LabelledArray
            The labelled array instance to test
        Tuple[elements, shape, labels, strides]
            A tuple of reference quantities used in constructing the 
            `LabelledArray` instance. These should be used to test correctness
            of the array instance.
        """
        l, m, n = 2, 3, 4
        shape = (l, m, n)
        labels = (('a', 'b'), ('a', 'b', 'c'), ('a', 'b', 'c', 'd'))

        strides = strides_from_shape(shape)

        elements = string.ascii_lowercase[:math.prod(shape)]
        elements = [char for char in elements]

        return LabelledArray(elements, shape, labels), (elements, shape, labels, strides)

    def test_shape(self, setup_array):
        """
        Test the `LabelledArray` instance has the correct shape
        """
        array, (_, ref_f_shape, *_) = setup_array
        # print(f"test has shape {array.shape} and vals {array.array}")
        assert array.shape == squeeze_shape(ref_f_shape)

    def test_f_shape(self, setup_array):
        """
        Test the `LabelledArray` instance has the correct (full) shape
        """
        array, (_, ref_f_shape, *_) = setup_array
        # print(f"test has shape {array.shape} and vals {array.array}")
        assert array.f_shape == ref_f_shape

    def test_ndim(self, setup_array):
        """
        Test `LabelledArray.ndim` is correct
        """
        array, (_, ref_f_shape, *_) = setup_array
        assert array.ndim == len(ref_f_shape)

    def test_f_ndim(self, setup_array):
        """
        Test `LabelledArray.f_ndim` is correct
        """
        array, (_, ref_f_shape, *_) = setup_array
        assert array.f_ndim == len(squeeze_shape(ref_f_shape))

    def test_single_elem_index(self, setup_array):
        """
        Test indexing a single element from a `LabelledArray`

        Note that the test for label indices won't work if the array instance
        doesn't have labels.
        """
        array, (ref_data, *_, ref_strides) = setup_array

        # Loop through each single index and check that the right element is selected
        # TODO: This won't work for arrays with axis labels
        all_axis_int_indices = [range(axis_size) for axis_size in array.shape]
        all_axis_str_indices = [axis_labels for axis_labels in array.labels]
        for mindex_int, mindex_str in zip(
                product(*all_axis_int_indices), product(*all_axis_str_indices)
            ):

            assert array[mindex_int] == ref_data[flat_idx(mindex_int, ref_strides)]
            assert array[mindex_str] == ref_data[flat_idx(mindex_int, ref_strides)]

    def test_multi_elem_index(self, setup_array):
        """
        Test multiple elements from a `LabelledArray`
        """
        array, _ = setup_array
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

    def test_generic_index(self, setup_array):
        """
        Test a generic multi-index
        """
        array, (elements, shape, *_) = setup_array

        mlabel_to_idx = array._MULTI_LABEL_TO_IDX
        test_idx = (0, 0)
        assert self._test_midx(test_idx, array, elements, shape, mlabel_to_idx)


    @staticmethod
    def _test_midx(
            midx: MultiGenIndex,
            array: LabelledArray[T],
            elements: FlatArray[T],
            shape: Shape,
            mlabel_to_idx: MultiLabelToStdIndex
        ):
        """
        Test a multi-index returns the correct values

        This compares the result of `array[midx]` against the results of
        indexing the reference data `(elements, shape, mlabel_to_idx)`.
        """
        ## Compute the indexed result from the reference data
        # Compute a flat reference index to get the 'correct' elements
        def require_list(x):
            if isinstance(x, (list, tuple)):
                return x
            else:
                return [x]

        midx = expand_multi_gen_idx(midx, shape)
        ref_midx = conv_multi_gen_to_std_idx(midx, shape, mlabel_to_idx)
        _ref_midx = [require_list(x) for x in ref_midx]

        strides = strides_from_shape(shape)
        def to_flat(midx):
            return np.sum(np.multiply(midx, strides))

        ref_idx_elements = [
            elements[to_flat(midx)] for midx in product(*_ref_midx)
        ]

        ## Compute the indexed result from the `LabelledArray`
        array_idx_elements = array[midx].array.tolist()

        ## Compare the reference and `LabelledArray` results to check
        return ref_idx_elements == array_idx_elements

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
@pytest.fixture(
    params=[
        ( (..., slice(None)), 1, (slice(None),) ),
        ( (slice(None), ...), 1, (slice(None),) ),

        ( (..., slice(None)), 4, (slice(None),)*4 ),
        ( (slice(None), ...), 4, (slice(None),)*4 ),

        ( (..., 3), 4, (slice(None),)*3 + (3,) ),
        ( (3, ...), 4, (3,) + (slice(None),)*3 ),

        ( (..., slice(None), 3), 4, (slice(None),)*2 + (slice(None), 3) ),
        ( (slice(None), 3, ...), 4, (slice(None), 3) + (slice(None),)*2 ),
    ]
)
def setup_idx(request):
    """
    Return and index and the correct 'expanded' index
    """
    idx, ndim, expanded_idx = request.param
    return idx, ndim, expanded_idx

def test_expand_multidx(setup_idx):
    """
    Test expansion of a multi-index with ellipses and/or missing axes
    """
    midx, ndim, ref_midx = setup_idx
    assert la.expand_multi_gen_idx(midx, ndim) == ref_midx

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
