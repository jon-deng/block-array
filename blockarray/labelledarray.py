"""
A multi-dimensional array with labelled indices

A `LabelledArray` is a multidimensional array of a fixed shape containing
arbitrary objects and with labelled indices in addition to integer indices
along each axis. These can be indexed in a similar way to `numpy.ndarray`.
"""

from typing import Optional, Union, List, Tuple, Generic
from itertools import chain
import math

import numpy as np

from .typing import (
    T,
    NestedArray,
    FlatArray,
    Shape,
    Strides,
    MultiLabels,

    GenIndex,
    StdIndex,

    MultiStdIndex,
    MultiGenIndex,

    LabelToStdIndex,
    MultiLabelToStdIndex
)

# TODO: Inconsistent types for 'list of integer' indexes;
# some functions use, tuples or lists, etc...
# Think of standard way to allow list of integer indices

def flatten_array(array: NestedArray[T]) -> Tuple[FlatArray[T], Shape]:
    """
    Return a flat array and shape from a nested array

    Parameters
    ----------
    array :
        A nested array

    Returns
    -------
    flat_array :
        The flattened array
    shape :
        The shape of the nested array
    """
    def check_is_nested(array):
        """
        Check whether an array is nested
        """
        # Checks whether each element of an array is another array
        is_array = [isinstance(elem, (list, tuple)) for elem in array]
        is_array_count = is_array.count(True)
        if is_array_count == len(is_array):
            # Make sure the nested sizes are correct
            assert all([len(elem) == len(array[0]) for elem in array])
            return True
        elif is_array_count == 0:
            return False
        else:
            raise ValueError("Improperly nested array")

    flat_array = array
    shape = (len(flat_array),)
    while check_is_nested(flat_array):
        shape += (len(flat_array[0]),)
        flat_array = [elem for elem in chain(*flat_array)]

    return flat_array, shape

def nest_array(array: FlatArray[T], strides: Strides) -> NestedArray[T]:
    """
    Return a nested array from a flat array and strides

    Parameters
    ----------
    array: Tuple, list
        A flat array
    strides:
        A tuple of strides

    Returns
    -------
    ret_array:
        The nested array
    """
    size = len(array)
    for stride in strides:
        assert math.remainder(size, stride) == 0
    # the last axis stride should be 1 for c-order
    assert strides[-1] == 1

    if len(strides) == 1:
        return array
    else:
        stride = strides[0]
        ret_array = [
            nest_array(array[ii*stride:(ii+1)*stride], strides[1:])
            for ii in range(size//stride)
        ]
        return ret_array

def validate_shape(array: FlatArray[T], shape: Shape):
    """
    Validate a flat array can be reshaped into a given shape

    Parameters
    ----------
    array :
        The flat array
    shape :
        The shape to validate `array` against

    Raises
    ------
    ValueError
        Raises `ValueError` if `shape` and `array` are not compatible
    """
    # use `abs()` as hacky way to account for collapsed dimensions (size -1)
    if len(array) != abs(math.prod(shape)):
        raise ValueError(f"`shape` {shape} is incompatible with array of length {len(array)}")

def validate_labels(labels: MultiLabels, shape: Shape):
    """
    Validate if n-dimensional labels and shape are compatible

    This checks that for each dimension:
        - There is one label for each index
        - there are no duplicate labels
        - or, there are no labels
        - if the dimension is collapsed, no labels are supplied

    Parameters
    ----------
    labels :
        The n-dimensional labels
    shape :
        The shape to validate `labels` against

    Raises
    ------
    ValueError
        Raises `ValueError` if `labels` and `shape` are not compatible
    """
    # Check that the number of dimensions in labels and shape is compatible
    if len(labels) != len(shape):
        raise ValueError(f"{len(labels)} axis labels is incompatible for array with {len(shape)} dimensions")

    for dim, (axis_labels, axis_size) in enumerate(zip(labels, shape)):
        if axis_size == -1:
            # Check that collapsed dimensions should have no labels
            if axis_labels != ():
                raise ValueError(f"Invalid non-empty axis labels {axis_labels} for reduced axis {dim}")
        else:
            if len(axis_labels) != 0:
                # Check that there is one label for each index along an axis
                if len(axis_labels) != axis_size:
                    raise ValueError(f"Invalid {len(axis_labels)} axis labels for axis {dim} with size {axis_size}")

                # Check that axis labels are unique
                if len(set(axis_labels)) != len(axis_labels):
                    raise ValueError(f"Invalid duplicate labels for axis {dim} with labels {axis_labels}")

def validate_gen_index_range(
        idx: GenIndex,
        label_to_idx: LabelToStdIndex,
        size: int
    ):
    """
    Validate an index selects a correct number of elements

    This checks that `idx`:
        - doesn't select duplicate elements
        - doesn't select more elements than the size of axis

    Parameters
    ----------
    idx :
        The index to check
    label_to_idx :
        A mapping from string labels to integer indices
    size :
        The size of the axis

    Raises
    ------
    ValueError :
        Raises `ValueError` if the index selects an incorrect number of elements
    """
    if isinstance(idx, list):
        _idx = conv_gen_to_std_idx(idx, label_to_idx, size)
        if len(set(_idx)) != len(_idx):
            raise ValueError(f"{idx} selects the same element multiple times")
        elif len(_idx) > size:
            raise ValueError(f"{idx} selects {len(idx)} elements for an axis of size {size:d}")

def validate_multi_gen_index_range(
        midx: MultiGenIndex,
        multi_label_to_idx: MultiLabelToStdIndex,
        shape: Shape
    ):
    """
    Validate a multi index selects a correct number of elements

    See `validate_gen_index_range` for further details
    """
    for idx, label_to_idx, size in zip(midx, multi_label_to_idx, shape):
        validate_gen_index_range(idx, label_to_idx, size)

class LabelledArray(Generic[T]):
    """
    An N-dimensional array with (optionally) labelled indices

    Parameters
    ----------
    array :
        A list/tuple/np.ndarray of items in the array. This is a flat list which
        is interpreted as an n-d array according to the supplied `shape` and a
        'C' ordering.
    shape :
        A tuple of axis sizes (n, m, ...), where axis 0 has size n, axis 1 has
        size m, etc. An axis size can also be -1 in which case the axis cannot
        be indexed but increases the number of dimensions of the array.
    labels :
        An optional tuple of labels corresponding to each index along an axis.
        `labels[0]` should contain the labels for indices along axis 0,
        `labels[1]` the indices along axis 1, etc. If a dimension of `labels` is
        an empty tuple, that dimension will not allow indexing by label.

    Attributes
    ----------
    array :
        A `np.ndarray` object array containing the elements
    shape :
        The N-d layout of the elements. For example, a shape `(2, 3)` represents
        an array of 2 elements in dimension 0 by 3 elements in dimension 1.
    labels :
        A nested tuple containing labels for each axis
    multi_label_to_idx :
        A mapping of labels to indices for each axis
    """

    def __init__(
            self,
            array: Union[FlatArray[T], np.ndarray],
            shape: Shape,
            labels: Optional[MultiLabels]=None
        ):
        # If no labels are supplied, use empty label tuples for each axis
        if labels is None:
            labels = ((),)*len(shape)
        else:
            # Convert any lists to tuples in labels
            labels = tuple(tuple(dim_labels) for dim_labels in labels)

        # Validate the array shape and labels
        validate_labels(labels, shape)
        validate_shape(array, shape)

        # Assign basic data
        self._shape = tuple(shape)
        self._labels = tuple(labels)
        if isinstance(array, np.ndarray):
            self._array = array.reshape(self.shape)
        else:
            self._array = np.ndarray(self.shape, object)
            self._array.reshape(-1)[:] = array

        self._MULTI_LABEL_TO_IDX = tuple(
            {label: ii for label, ii in zip(axis_labels, idxs)}
            for axis_labels, idxs in zip(self.labels, [range(axis_size) for axis_size in self.shape])
        )

    @property
    def array(self) -> np.ndarray:
        """Return the numpy object array container"""
        return self._array

    @property
    def f_shape(self) -> Shape:
        """Return the array shape"""
        return self._shape

    @property
    def f_ndim(self) -> int:
        """Return the number of dimensions (number of axes)"""
        return len(self.f_shape)

    @property
    def ndim(self) -> int:
        """Return the reduced number of dimensions (number of axes)"""
        return len(self.shape)

    @property
    def f_dims(self) -> Tuple[int, ...]:
        """Return the axis/dimensions indices"""
        return tuple(range(self.f_ndim))

    @property
    def dims(self) -> Tuple[int, ...]:
        """Return the reduced axis/dimensions indices"""
        return tuple(ii for ii, ax_size in zip(self.f_dims, self.f_shape) if ax_size != -1)

    @property
    def f_labels(self) -> MultiLabels:
        """Return the array labels"""
        return self._labels

    @property
    def shape(self) -> Shape:
        """Return the reduced array shape"""
        ret_rshape = [axis_size for axis_size in self.f_shape if axis_size != -1]
        return tuple(ret_rshape)

    @property
    def labels(self) -> MultiLabels:
        """Return the reduced labels"""
        ret_rlabels = [
            axis_labels
            for axis_labels, axis_size in zip(self.f_labels, self.f_shape)
            if axis_size > 0
        ]
        return tuple(ret_rlabels)

    @property
    def size(self) -> int:
        """Return the array size"""
        return math.prod(self.shape)

    def __len__(self):
        return self.size

    def __getitem__(self, multi_idx) -> Union[T, 'LabelledArray[T]']:
        # This ensures that single axis indices (i.e. `x[1]`, `x[:]`) are
        # converted to a size 1 tuple; the indexing functions all expect
        # multi index tuples
        multi_idx = (multi_idx,) if not isinstance(multi_idx, tuple) else multi_idx
        multi_idx = expand_multi_gen_idx(multi_idx, self.shape)
        validate_multi_gen_index_range(multi_idx, self._MULTI_LABEL_TO_IDX, self.shape)
        multi_idx = conv_multi_gen_to_std_idx(multi_idx, self.shape, self._MULTI_LABEL_TO_IDX)

        ## Find the returned `BlockArray` shape and labels
        # Get the reduced/collapsed shape + labels (these are collapsed since
        # indexing implictly occurs along non collapsed dims)
        def _ax_size_from_idx(axis_idx):
            if isinstance(axis_idx, list):
                return len(axis_idx)
            elif isinstance(axis_idx, int):
                return -1
            else:
                assert False
        ret_shape = tuple(_ax_size_from_idx(axis_idx) for axis_idx in multi_idx)

        def _ax_labels_from_idx(axis_idx, ax_labels):
            if isinstance(axis_idx, list):
                if ax_labels != ():
                    return tuple(ax_labels[ii] for ii in axis_idx)
                else:
                    return ()
            elif isinstance(axis_idx, int):
                return ()
            else:
                assert False
        ret_labels = tuple(
            _ax_labels_from_idx(axis_idx, axis_labels)
            for axis_labels, axis_idx in zip(self.labels, multi_idx)
        )

        # Splice the non collapsed shape + labels into the corresponding 'full'
        # tuples
        f_shape = [-1] * self.f_ndim
        f_labels = [()] * self.f_ndim
        for ii, ax_size, ax_labels in zip(self.dims, ret_shape, ret_labels):
            f_shape[ii] = ax_size
            f_labels[ii] = ax_labels
        f_shape = tuple(f_shape)
        f_labels = tuple(f_labels)

        ## Index the subset of elements from the array
        # enclose single ints in a list so it works with itertools
        ndim = len(multi_idx)
        midx = [[idx] if isinstance(idx, int) else idx for idx in multi_idx]
        midx = [
            np.array(idx, dtype=np.intp)[(slice(None),)+(None,)*n]
            for n, idx in zip(range(ndim-1, -1, -1), midx)
        ]
        midx = np.broadcast_arrays(*midx)
        ret_array = self._array[tuple(midx)].reshape(-1)

        if f_shape == (-1,) * self.f_ndim:
            assert len(ret_array) == 1
            return ret_array[0]
        else:
            return LabelledArray(ret_array, f_shape, f_labels)

    ## Copy methods
    def copy(self):
        """Return a copy of the array"""
        ret_labels = self.f_labels
        ret_shape = self.f_shape
        ret_array = [elem.copy() for elem in self.array.flat]
        return self.__class__(ret_array, ret_shape, ret_labels)

    def __copy__(self):
        return self.copy()

    ## Dict like interface, over the first axis
    def __contains__(self, key):
        return key in self._MULTI_LABEL_TO_IDX[0]

    def items(self):
        """Return an iterable of 'key: value' pairs along the first axis"""
        return zip(self.labels[0], self)

    ## Iterable interface over the first axis
    def __iter__(self) -> Union[List['LabelledArray'], List[T]]:
        for ii in range(self.shape[0]):
            yield self[ii]

# For the below, the naming convention where applicable is:
# dnote multi-indexes by `multidx`
# use `gen_` and `std_` to denote general and standard indexes
def expand_multi_gen_idx(
        multidx: MultiGenIndex, shape: Shape
    ) -> MultiGenIndex:
    """
    Expands missing axis indices and/or ellipses in a general multi-index

    This ensures the number of axis indices in a general multi index
    matches the total number of axes.

    Parameters
    ----------
    multi_idx: tuple(GeneralIndex)
        A tuple of general indices used to index individual axes
    shape: tuple(int)
        The shape of the array being indexed
    """
    # Check that there are fewer dimensions indexed than number of dimensions
    assert len(multidx) <= len(shape)

    num_ellipse = multidx.count(...)
    assert num_ellipse <= 1

    if num_ellipse == 1:
        num_ax_expand = len(shape) - len(multidx) + 1
        axis_expand = multidx.index(...)
    else:
        num_ax_expand = len(shape) - len(multidx)
        axis_expand = len(multidx)

    new_multi_gidx = (
        multidx[:axis_expand]
        + tuple(num_ax_expand*[slice(None)])
        + multidx[axis_expand+num_ellipse:]
        )
    return new_multi_gidx

# This function handles conversion of any of the general index/indices
# to a standard index/indices
def conv_multi_gen_to_std_idx(
        multidx: MultiGenIndex,
        shape: Shape,
        multi_label_to_idx: MultiLabelToStdIndex
    ) -> MultiStdIndex:
    """
    Return a standard multi-index from a general multi-index

    The standard multi-index has the type Tuple[Union[Int, Tuple[int, ...]], ...].
    In other words it's a tuple with each element being either a single int, or an
    iterable of ints, representing the indexes being selected from the given axis.

    Parameters
    ----------
    multi_idx: tuple(GeneralIndex)
        A tuple of general indices used to index individual axes
    shape: tuple(int)
        The shape of the array being indexed
    multi_label_to_idx:
        A tuple of mappings, where each mapping contains the map from label to index
        for the given axis
    """
    multi_sidx = [
        conv_gen_to_std_idx(index, axis_label_to_idx, axis_size)
        for index, axis_size, axis_label_to_idx
        in zip(multidx, shape, multi_label_to_idx)
    ]
    # Sanity check
    for idx in multi_sidx:
        assert isinstance(idx, (List, int))
    return tuple(multi_sidx)

def conv_gen_to_std_idx(
        idx: GenIndex,
        label_to_idx: LabelToStdIndex,
        size: int
    ) -> StdIndex:
    """
    Return a standard index(s) form any of the 3 valid general index formats

    Parameters
    ----------
    idx : str
        A general index
    label_to_idx : Dict
        Mapping from label indices to corresponding integer indices
    size : int
        Size of the iterable
    """
    assert len(label_to_idx) == size or len(label_to_idx) == 0

    if isinstance(idx, slice):
        return conv_slice_to_std_idx(idx, size)
    elif isinstance(idx, (list, tuple)):
        return conv_list_to_std_idx(idx, label_to_idx, size)
    elif isinstance(idx, str):
        return conv_label_to_std_idx(idx, label_to_idx, size)
    elif isinstance(idx, int):
        return conv_neg_to_std_idx(idx, size)
    else:
        raise TypeError(f"Unknown index {idx} of type {type(idx)}.")

# These functions convert general indices (GeneralIndex) to standard indices
# (StandardIndex)
def conv_list_to_std_idx(
        idx: Union[List[Union[str, int]], Tuple[Union[str, int]]],
        label_to_idx: LabelToStdIndex,
        size: int
    ) -> StdIndex:
    """
    Return a sequence of indices so that each index is a positive integer

    Parameters
    ----------
    idx :
        A list of either string or integer indices (can be negative)
    label_to_idx :
        A mapping from string to integer indices
    size :
        The size of the axis

    Returns
    -------
        A list of positive integer indices
    """
    return [
        conv_label_to_std_idx(ii, label_to_idx, size)
        if isinstance(ii, str) else conv_neg_to_std_idx(ii, size)
        for ii in idx
    ]

def conv_slice_to_std_idx(idx: slice, size: int) -> StdIndex:
    """
    Return a sequence of postiive indices from a slice

    Parameters
    ----------
    idx :
        A list of either string or integer indices (can be negative)
    size :
        The size of the axis

    Returns
    -------
        A list of positive integer indices
    """
    start = conv_slice_start_to_idx(idx.start, size)
    stop = conv_slice_stop_to_idx(idx.stop, size)
    if idx.step is None:
        step = 1
    else:
        step = idx.step
    return list(range(start, stop, step))

# These functions convert a general single index (GeneralIndex)
# to a standard single index (StandardIndex, specifically StdIndex)
def conv_label_to_std_idx(
        idx: str,
        label_to_idx: LabelToStdIndex,
        size: int
    ) -> StdIndex:
    """
    Return the integer index corresponding to a labelled index

    Parameters
    ----------
    idx : str
        Label index of an element
    label_to_idx : Dict
        Mapping from label indices to corresponding integer indices
    size : int
        Size of the iterable
    """
    ret_index = label_to_idx[idx]
    assert ret_index >= 0 and ret_index < size
    return ret_index

def conv_neg_to_std_idx(idx: int, size: int) -> StdIndex:
    """
    Return the positive index corresponding to a negative index

    For an array of size N, a negative index `-k`, correspond to the positive
    index `N-k`.

    Parameters
    ----------
    idx : int
        Index of the element
    size : int
        Number of elements in the iterable
    """
    if idx >= 0:
        return idx
    else:
        return size+idx

def conv_slice_start_to_idx(idx: Union[int, None], size: int) -> StdIndex:
    """
    Return an int representing the starting index from a slice object

    Parameters
    ----------
    idx : int
        Index of the element representing a slice objects `.start` attribute
    size : int
        Size of the iterable
    """
    if idx is None:
        return 0
    else:
        return conv_neg_to_std_idx(idx, size)

def conv_slice_stop_to_idx(idx: Union[int, None], size: int) -> StdIndex:
    """
    Return an int representing the end index from a slice object

    Parameters
    ----------
    idx : int
        Index of the element representing a slice objects `.stop` attribute
    size : int
        Size of the iterable
    """
    if idx is None:
        return size
    else:
        return conv_neg_to_std_idx(idx, size)
