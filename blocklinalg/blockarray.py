"""
A BlockArray is a multidimensional array of a fixed shape (similar to numpy arrays) containing
arbitrary objects and where axis index has a label
"""

from typing import TypeVar, Tuple, Union, Mapping
from itertools import product, chain

import numpy as np
import math
import functools as ft

T = TypeVar("T")
NestedArray = Tuple['NestedArray', ...]
Array = Tuple[T, ...]
Shape = Tuple[int, ...]
Labels = Tuple[str, ...]
AxisBlockLabels = Tuple[Tuple[str, ...], ...]

Index = int
Indices = Tuple[int, ...]
EllipsisType = type(...)

GeneralIndex = Union[slice, Index, str, Indices, EllipsisType]
StandardIndex = Union[Index, Indices]

def block_array(array, labels):
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
    return BlockArray(flat_array, shape, labels)

def flatten_array(array):
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
        print(flat_array)
        shape.append(len(flat_array))
        flat_array = [elem for elem in chain(*flat_array)]

    shape.append(len(flat_array)//math.prod(shape))

    return flat_array, tuple(shape)


class BlockArray:
    """
    An N-dimensional array

    Parameters
    ----------
    array : tuple of objects, length N
    shape : tuple of ints, length N
    """

    def __init__(self, array: Array, shape: Shape, labels: AxisBlockLabels):
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
            {label: ii for label, ii in zip(axis_labels, idxs)} 
            for axis_labels, idxs in zip(self.labels, [range(axis_size) for axis_size in self.shape])])

    @property
    def array(self):
        """
        Return raw array
        """
        return self._array

    @property
    def shape(self):
        """
        Return array shape
        """
        return self._shape

    @property
    def labels(self):
        """
        Return array axis block labels
        """
        return self._labels

    @property 
    def size(self):
        """
        Return array size
        """
        return math.prod(self.shape)

    def __len__(self):
        return self.size

    def __getitem__(self, multi_idx):
        multi_idx = convert_general_multi_idx(multi_idx, self.shape, self._MULTI_LABEL_TO_IDX)

        # Find the returned BlockArray's shape and labels
        ret_shape = tuple([len(axis_idxs) for axis_idxs in multi_idx if isinstance(axis_idxs, tuple)])
        ret_labels = tuple([
            tuple([axis_labels[ii] for ii in axis_idxs])
            for axis_labels, axis_idxs in zip(self.labels, multi_idx) if isinstance(axis_idxs, tuple)
        ])

        # enclose single ints in a list so it works with itertools
        multi_idx = [(idx,) if isinstance(idx, int) else idx for idx in multi_idx]
        ret_flat_idxs = [to_flat_idx(idx, self._STRIDES) for idx in product(*multi_idx)]

        ret_array = tuple([self.array[flat_idx] for flat_idx in ret_flat_idxs])

        if ret_shape == ():
            assert len(ret_array) == 1
            return ret_array[0]
        else:
            return BlockArray(ret_array, ret_shape, ret_labels)

    ## Copy methods
    def copy(self):
        """Return a copy"""
        ret_labels = self.labels
        ret_shape = self.shape
        ret_array = [elem.copy() for elem in self.array]
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
        for ii in range(self.shape[0]):
            yield self[ii]


def to_flat_idx(multi_idx: StandardIndex, strides: Tuple[int, ...]) -> Union[Index, Indices]:
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

def expand_multi_idx(multi_idx: GeneralIndex, shape: Shape) -> GeneralIndex:
    """
    Expands missing axis indices or ellipses in a general multi-index

    This ensures the number of axis indices in a general multi index
    matches the total number of axes.

    Parameters
    ----------
    multi_idx: tuple(GeneralIndex)
        A tuple of general indices used to index individual axes
    shape: tuple(int)
        The shape of the array being indexed
    """
    num_ellipse = multi_idx.count(...)
    assert num_ellipse <= 1

    if num_ellipse == 1:
        num_missing_axis_idx = len(shape) - len(multi_idx) + 1
        axis_expand = multi_idx.index(...)
    else:
        num_missing_axis_idx = len(shape) - len(multi_idx)
        axis_expand = len(multi_idx)
    new_multi_idx = tuple(
        list(multi_idx[:axis_expand])
        + num_missing_axis_idx*[slice(None)]
        + list(multi_idx[axis_expand+num_ellipse:])
        )
    return new_multi_idx

def convert_general_multi_idx(
    multi_idx: Tuple[GeneralIndex, ...], 
    shape: Shape, 
    multi_label_to_idx: Tuple[Mapping[str, int], ...]) -> StandardIndex:
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
    multi_idx = (multi_idx,) if not isinstance(multi_idx, tuple) else multi_idx

    multi_idx = expand_multi_idx(multi_idx, shape)
    out_multi_idx = [
        convert_general_idx(index, axis_size, axis_label_to_idx) 
        for index, axis_size, axis_label_to_idx in zip(multi_idx, shape, multi_label_to_idx)]
    return tuple(out_multi_idx)

def convert_general_idx(idx: GeneralIndex, size, label_to_idx) -> Union[Indices, Index]:
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
        return tuple(
            [convert_label_idx(ii, label_to_idx, size) if isinstance(ii, str) else ii
            for ii in idx]
            )
    elif isinstance(idx, str):
        return convert_label_idx(idx, label_to_idx, size)
    elif isinstance(idx, int):
        return convert_neg_idx(idx, size)
    else:
        raise TypeError(f"Unknown index {idx} of type {type(idx)}.")

# The below functions convert general 1D slice indices to standard 1D slice indices, 
# tuples of positive integers
def convert_slice(idx: slice, size: int) -> Indices:
    """
    Return the sequence of indexes corresponding to a slice
    """
    start = convert_start_idx(idx.start, size)
    stop = convert_stop_idx(idx.stop, size)
    if idx.step is None:
        step = 1
    else:
        step = idx.step
    return tuple(range(start, stop, step))

# The below functions convert general single indexes to a standard single index, a positive integer
def convert_label_idx(idx: str, label_to_idx: Mapping[str, int], size: int) -> int:
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

def convert_neg_idx(idx: int, size: int) -> int:
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
        return size-idx

def convert_start_idx(idx: Union[int, None], size: int) -> int:
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

def convert_stop_idx(idx: Union[int, None], size: int) -> int:
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
    print(f"test[:, :, :] has shape {test[:, :, :].shape} and vals {test[:, :, :].array}")
    print(f"test[:] has shape {test[:].shape} and vals {test[:].array}")

    print(flatten_array([[1, 2, 3], [4, 5, 6]]))
    