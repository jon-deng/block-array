"""
A LabelledArray is a multidimensional array of a fixed shape containing
arbitrary objects and with labelled indices along each axis. These can be
indexed in a similar way to `numpy.ndarray`.
"""

from typing import TypeVar, Tuple, Union, Mapping, Optional
from itertools import product, chain, accumulate

import math

T = TypeVar("T")
NestedArray = Tuple['NestedArray', ...]
FlatArray = Tuple[T, ...]
Shape = Tuple[int, ...]
Strides = Tuple[int, ...]
Labels = Tuple[str, ...]
AxisBlockLabels = Tuple[Tuple[str, ...], ...]

IntIndex = int
IntIndices = Tuple[int, ...]
EllipsisType = type(...)

GeneralIndex = Union[slice, IntIndex, str, IntIndices, EllipsisType]
StandardIndex = Union[IntIndex, IntIndices]

MultiStandardIndex = Tuple[StandardIndex, ...]
MultiGeneralIndex = Tuple[GeneralIndex, ...]

LabelToIntIndex = Mapping[str, IntIndex]
MultiLabelToIntIndex = Tuple[LabelToIntIndex, ...]

def block_array(array: FlatArray, labels: Labels):
    """
    Return a BlockArray from nested lists/tuples

    Parameters
    ----------
    array : nested tuple/lists
        Nested list representation of an nd array
    labels : AxisBlockLabels
        list of labels for each axis index. The number of labels along each axis
        should match the size of that axis
    """
    flat_array, shape = flatten_array(array)
    return LabelledArray(flat_array, shape, labels)

def flatten_array(array: NestedArray):
    """
    Flattens and return the shape of a nested array
    """

    def check_is_nested(array):
        elem_is_array = [isinstance(elem, (list, tuple)) for elem in array]
        is_array_count = elem_is_array.count(True)
        if is_array_count == len(elem_is_array):
            # Make sure the nested sizes are correct
            assert all([len(elem) == len(array[0]) for elem in array])
            return True
        elif is_array_count == 0:
            return False
        else:
            raise ValueError("Improperly nested array")

    flat_array = array
    shape = []
    while check_is_nested(flat_array):
        shape.append(len(flat_array))
        flat_array = [elem for elem in chain(*flat_array)]

    shape.append(len(flat_array)//math.prod(shape))

    return flat_array, tuple(shape)

def nest_array(array: FlatArray, strides: Strides):
    """
    Convert a flat array into a nested array from given strides

    Parameters
    ----------
    array: Tuple, list
        A flat array
    strides
        A tuple of strides
    """
    size = len(array)
    for stride in strides:
        assert math.remainder(size, stride) == 0
    assert strides[-1] == 1 # the last axis stride should be 1 for c-order

    if len(strides) == 1:
        return array
    else:
        stride = strides[0]
        ret_array = [
            nest_array(array[ii*stride:(ii+1)*stride], strides[1:])
            for ii in range(size//stride)
            ]
        return ret_array

def validate_shape(array, shape):
    """Validates the array shape"""
    # use `abs()` as hacky way to account for reduced dimensions represented
    # by -1
    if len(array) != abs(math.prod(shape)):
        raise ValueError(f"shape {shape} is incompatible with array of length {len(array)}")

def validate_labels(labels, shape):
    """Validates the array labels"""
    if len(labels) != len(shape):
        raise ValueError(f"{len(labels)} axis labels is incompatible for array with {len(shape)} dimensions")

    for dim, (axis_labels, axis_size) in enumerate(zip(labels, shape)):
        if axis_size == -1:
            if axis_labels != ():
                raise ValueError(f"Found non-empty axis labels {axis_labels} for reduced axis {dim}")
        else:
            # Check that there is one label for each index along an axis
            if len(axis_labels) != axis_size:
                raise ValueError(f"{len(axis_labels)} axis labels is incompatible for axis {dim} with size {axis_size}")

            # Check that axis labels are unique
            if len(set(axis_labels)) != len(axis_labels):
                raise ValueError(f"duplicate labels found for axis {dim} with labels {axis_labels}")

def validate_general_idx(idx, size):
    """Validate a general index"""
    lb = -size
    ub = size-1
    def valid_index(idx, lb, ub):
        """Whether an integer index is valid"""
        return (idx<=ub and idx>=lb)

    if isinstance(idx, slice):
        start, stop = idx.start, idx.stop
        if start is not None:
            if not valid_index(start, lb, ub):
                raise IndexError(f"slice start index {start} out of range for axis of size {size}")

        if stop is not None:
            if not valid_index(stop, lb-1, ub+1):
                # The stop index is noninclusive so goes +1 off the valid index bound
                raise IndexError(f"slice stop index {stop} out of range for axis of size {size}")
    elif isinstance(idx, int):
        if not valid_index(idx, lb, ub):
            raise IndexError(f"index {idx} out of range for axis of size {size}")
    elif isinstance(idx, (list, tuple)):
        valid_idxs = [valid_index(ii, lb, ub) for ii in idx if isinstance(ii, int)]
        if not all(valid_idxs):
            raise IndexError(f"index out of range in {idx} for axis of size {size}")

def validate_multi_general_idx(multi_idx: MultiGeneralIndex, shape: Shape):
    """Validate a multi general index"""
    for idx, size in zip(multi_idx, shape):
        validate_general_idx(idx, size)


class LabelledArray:
    """
    An N-dimensional array with labelled indices

    Parameters
    ----------
    array
        A list of items in the array. This is a flat list which is interpreted
        with the supplied shape according to 'C' ordering.
    shape
        A tuple of axis sizes (n, m, ...), where axis 0 has size n, axis 1 has
        size m, etc.
    labels
        A tuple of labels corresponding each index along an axis. `labels[0]`
        should contain the labels for indices along axis 0, `labels[1]` the
        indices along axis 1, etc.
    """

    def __init__(self, array: FlatArray, shape: Shape, labels: Optional[AxisBlockLabels]=None):
        if labels is None:
            labels = tuple([tuple([str(ii) for ii in range(axis_size)]) for axis_size in shape])
        # Convert any lists to tuples in labels
        labels = tuple([tuple(dim_labels) for dim_labels in labels])

        # Validate the array shape and labels
        validate_labels(labels, shape)
        validate_shape(array, shape)

        # Assign basic data
        self._array = tuple(array)
        self._shape = tuple(shape)
        self._labels = tuple(labels)

        # Compute convenience constants
        _strides = [
            stride for stride
            in accumulate(self.rshape[-1:0:-1], lambda a, b: a*b, initial=1)]
        self._STRIDES = tuple(_strides[::-1])
        self._MULTI_LABEL_TO_IDX = tuple([
            {label: ii for label, ii in zip(axis_labels, idxs)}
            for axis_labels, idxs in zip(self.rlabels, [range(axis_size) for axis_size in self.rshape])])

    @property
    def array_flat(self):
        """Return the flat array representation"""
        return self._array

    @property
    def array_nested(self):
        """Return a nested array representation"""
        return nest_array(self.array_flat, self._STRIDES)

    @property
    def shape(self):
        """Return the array shape"""
        return self._shape

    @property
    def ndim(self):
        """Return the number of dimensions (number of axes)"""
        return len(self.shape)

    @property
    def labels(self):
        """Return the array labels"""
        return self._labels

    @property
    def rshape(self):
        """
        Return the reduced array shape
        """
        ret_rshape = [axis_size for axis_size in self.shape if axis_size != -1]
        return tuple(ret_rshape)

    @property
    def rlabels(self):
        """
        Return the reduced labels
        """
        ret_rlabels = [axis_labels for axis_labels in self.labels if axis_labels != ()]
        return ret_rlabels

    @property
    def size(self):
        """Return the array size"""
        return math.prod(self.rshape)

    def __len__(self):
        return self.size

    def __getitem__(self, multi_idx):
        multi_idx = (multi_idx,) if not isinstance(multi_idx, tuple) else multi_idx
        multi_idx = expand_multi_idx(multi_idx, self.rshape)
        validate_multi_general_idx(tuple(multi_idx), self.rshape)

        multi_idx = convert_multi_general_idx(multi_idx, self.rshape, self._MULTI_LABEL_TO_IDX)

        # Find the returned BlockArray's shape and labels
        # -1 represents a reduced dimension,
        ret_shape = tuple([
            len(axis_idx) if isinstance(axis_idx, (list, tuple)) else -1
            for axis_idx in multi_idx
            ])
        ret_labels = tuple([
            (
                tuple([axis_labels[ii] for ii in axis_idx])
                if isinstance(axis_idx, (list, tuple))
                else ())
            for axis_labels, axis_idx in zip(self.labels, multi_idx)
        ])

        # enclose single ints in a list so it works with itertools
        multi_idx = [(idx,) if isinstance(idx, int) else idx for idx in multi_idx]
        ret_flat_idxs = [to_flat_idx(idx, self._STRIDES) for idx in product(*multi_idx)]

        ret_array = tuple([self.array_flat[flat_idx] for flat_idx in ret_flat_idxs])

        if ret_shape == (-1,) * len(ret_shape):
            assert len(ret_array) == 1
            return ret_array[0]
        else:
            return LabelledArray(ret_array, ret_shape, ret_labels)

    ## Copy methods
    def copy(self):
        """Return a copy of the array"""
        ret_labels = self.labels
        ret_shape = self.shape
        ret_array = [elem.copy() for elem in self.array_flat]
        return self.__class__(ret_array, ret_shape, ret_labels)

    def __copy__(self):
        return self.copy()

    ## Dict like interface, over the first axis
    def __contains__(self, key):
        return key in self._MULTI_LABEL_TO_IDX[0]

    def items(self):
        return zip(self.labels[0], self)

    ## Iterable interface over the first axis
    def __iter__(self):
        for ii in range(self.rshape[0]):
            yield self[ii]


def to_flat_idx(
    multi_idx: MultiStandardIndex, strides: Strides) -> StandardIndex:
    """
    Return a flat index given a multi-index and strides for each dimension

    Parameters
    ----------
    multi_idx: tuple(GeneralIndex)
        A tuple of general indices used to index individual axes
    strides: tuple(int)
        The integer offset for each axis according to c-ordering
    shape: tuple(int)
        The shape of the array being indexed
    """
    return sum([idx*stride for idx, stride in zip(multi_idx, strides)])

def expand_multi_idx(
    multi_idx: MultiGeneralIndex, shape: Shape) -> MultiGeneralIndex:
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
    assert len(multi_idx) <= len(shape)

    num_ellipse = multi_idx.count(...)
    assert num_ellipse <= 1

    if num_ellipse == 1:
        num_missing_axis_idx = len(shape) - len(multi_idx) + 1
        axis_expand = multi_idx.index(...)
    else:
        num_missing_axis_idx = len(shape) - len(multi_idx)
        axis_expand = len(multi_idx)

    new_multi_idx = (
        multi_idx[:axis_expand]
        + tuple(num_missing_axis_idx*[slice(None)])
        + multi_idx[axis_expand+num_ellipse:]
        )
    return new_multi_idx

# This function handles conversion of any of the general index/indices
# to a standard index/indices
def convert_multi_general_idx(
    multi_idx: MultiGeneralIndex,
    shape: Shape,
    multi_label_to_idx: MultiLabelToIntIndex) -> MultiStandardIndex:
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
    out_multi_idx = [
        convert_general_idx(index, axis_size, axis_label_to_idx)
        for index, axis_size, axis_label_to_idx in zip(multi_idx, shape, multi_label_to_idx)]
    return tuple(out_multi_idx)

def convert_general_idx(
    idx: GeneralIndex,
    size: int,
    label_to_idx: LabelToIntIndex) -> StandardIndex:
    """
    Return a standard index corresponding to any of the general index approaches

    Parameters
    ----------
    idx : str
        A general index
    label_to_idx : Dict
        Mapping from label indices to corresponding integer indices
    size : int
        Size of the iterable
    """
    assert len(label_to_idx) == size

    if isinstance(idx, slice):
        return convert_slice(idx, size)
    elif isinstance(idx, (list, tuple)):
        return [
            convert_label_idx(ii, label_to_idx, size) if isinstance(ii, str) else ii
            for ii in idx]
    elif isinstance(idx, str):
        return convert_label_idx(idx, label_to_idx, size)
    elif isinstance(idx, int):
        return convert_neg_idx(idx, size)
    else:
        raise TypeError(f"Unknown index {idx} of type {type(idx)}.")

# These functions convert general indices (GeneralIndex) to standard indices
# (StandardIndex)
def convert_slice(idx: slice, size: int) -> IntIndices:
    """
    Return the sequence of indexes corresponding to a slice
    """
    start = convert_start_idx(idx.start, size)
    stop = convert_stop_idx(idx.stop, size)
    if idx.step is None:
        step = 1
    else:
        step = idx.step
    return list(range(start, stop, step))

# These functions convert a general single index (GeneralIndex)
# to a standard single index (StandardIndex, specifically IntIndex)
def convert_label_idx(idx: str, label_to_idx: Mapping[str, int], size: int) -> IntIndex:
    """
    Return an integer index corresponding to a label

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

def convert_neg_idx(idx: int, size: int) -> IntIndex:
    """
    Return the index representing the equivalent negative index

    For an array of size N, a negative index `-k`, correspond to the positive
    index `N-k`.

    Parameters
    ----------
    idx : int
        Index of the element
    size : int
        Size of the iterable
    """
    if idx >= 0:
        return idx
    else:
        return size+idx

def convert_start_idx(idx: Union[int, None], size: int) -> IntIndex:
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
        return convert_neg_idx(idx, size)

def convert_stop_idx(idx: Union[int, None], size: int) -> IntIndex:
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
        return convert_neg_idx(idx, size)
