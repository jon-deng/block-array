"""
This module contains the block array definition and defines some basic operations
"""

from numbers import Number
from typing import TypeVar, Optional, Union, Callable, Generic, Tuple
from itertools import product, accumulate
import functools
import operator

import numpy as np

from . import labelledarray as larr
from . import subops as gops
from .typing import (
    BlockShape,
    FlatArray,
    Labels,
    Shape,
    MultiLabels,
    Scalar,
    MultiGenIndex,
    AxisSize,
)
from .misc import replace

T = TypeVar('T')
V = TypeVar('V')


## `BlockArray` object + core functions
class BlockArray(Generic[T]):
    """
    An n-dimensional block array

    `BlockArray` represents a block (or nested) array by storing sub-arrays
    correspondings to each block.

    Parameters
    ----------
    subarrays :
        The subarray elements. Depending on the type of array, the remaining parameters
        are interpreted differently:
            - If `array` is a `LabelledArray`, `shape` and `labels` parameters are
            not required.
            - If `array` is a nested list/tuple of subtensor elements, the shape will
            be derived from the shape of the nested list/tuple so the `shape` parameter
            is optional.
            - If `array` is a flat list/tuple of subtensor elements, a shape parameter
            must be supplied or the flat list will be interpreted as 1D nested list/tuple.
    shape :
        The shape of the blocks in the block tensor. For example, (2, 3) is a
        block matrix with 2 row blocks by 3 column blocks.
    labels :
        Labels for each block along each axis.
        If not provided, these default to string representation of integers.

    Attributes
    ----------
    size :
        The total number of subarrays contained in the block array.

    f_shape :
        The number of blocks (or subarrays) along each axis. For example, a
        block matrix with 2 row blocks and 3 column blocks has shape `(2, 3)`.

        A special size of -1 indicates a "reduced" block axis. Similar to
        an axis size of 1, an axis size of -1 contains 1 block; however,
        the dimension is implictly ignored for indexing.

        Similar to scalar `numpy` arrays, reduced axes are created by
        indexing a single element from the given axis (i.e. `[..., 0]` creates
        a reduced axis for the last axis). While reduced axes don't need to
        be tracked for scalar arrays, it is useful to track reduced axes for
        block arrays because reducing a block axis does not reduce corresponding
        subarray axes and the number of block and subarray dimensions should
        match.
    f_bshape :
        The 'block shape' (or nested shape) of the block array. This stores the
        axis sizes of subarrays along each axis. For example, a block shape
        `((120, 6), (5, 4))` represents a 2-by-2 block matrix with entries:
            - (0, 0) is a 120x5 matrix
            - (0, 1) is a 120x4 matrix
            - (1, 0) is a 6x5 matrix
            - (1, 1) is a 6x4 matrix

        A reduced axis contains a single integer representing its shape. For
        example, a block shape `((120, 6), (5, 4), 2)` represents a 2-by-2 block
        array with a reduced final axis and has entries:
            - (0, 0) is a 120x5x2 matrix
            - (0, 1) is a 120x4x2 matrix
            - (1, 0) is a 6x5x2 matrix
            - (1, 1) is a 6x4x2 matrix
    f_labels :
        A tuple of labels for each block along each axis
    f_ndim :
        The number of dimensions
    f_dims :
        A tuples of indices for each dimension

    shape, bshape :
        Collapsed versions of `f_shape` and `f_bshape` respectively. These
        have the same format as their correponding attributes but do not
        include any collapsed dimensions.
    labels :
        Labels for non-collapsed axes
    ndim :
        The number of non-collapsed dimensions
    dims :
        A tuple of indices for each non-collapsed dimension

    mshape :
        The shape of the block array's monolithic equivalent.

    array : np.ndarray
        The `np.ndarray` object array containing subarrays
    larray : larr.LabelledArray
        The `LabelledArray` instance used to store the subtensors in a block
        format
    """

    def __new__(
        cls,
        subarrays: Union[larr.LabelledArray[T], larr.NestedArray[T], larr.FlatArray[T]],
        shape: Optional[Shape] = None,
        labels: Optional[MultiLabels] = None,
        wrap: Callable[[T], gops.GenericSubarray[T]] = gops.wrap,
        check_bshape: bool = True,
    ):
        # Return a subarray instance if an explicit `shape` indicates all
        # block axes are collapsed
        # i.e. if `shape=(-1, -1, ..., -1)` then just return the single subarray
        if shape is None:
            return object.__new__(cls)
        elif shape == (-1,) * len(shape):
            # Get the flat list of subarrays and the shape to validate the shape
            assert len(subarrays) == 1
            return subarrays[0]
        else:
            return object.__new__(cls)

    def __init__(
        self,
        subarrays: Union[larr.LabelledArray[T], larr.NestedArray[T], larr.FlatArray[T]],
        shape: Optional[Shape] = None,
        labels: Optional[MultiLabels] = None,
        wrap: Callable[[T], gops.GenericSubarray[T]] = gops.wrap,
        check_bshape: bool = True,
    ):

        self._larray = larr.LabelledArray(
            *self._process_subarrays(subarrays, shape, labels, wrap=wrap)
        )
        self._bshape = _f_bshape_from_larray(self._larray)

        if check_bshape:
            _validate_f_bshape_from_larray(self._larray, self.f_bshape)

    @staticmethod
    def _process_subarrays(
        subarrays: Union[larr.LabelledArray[T], larr.NestedArray[T], larr.FlatArray[T]],
        shape: Optional[Shape] = None,
        labels: Optional[MultiLabels] = None,
        wrap: Callable[[T], gops.GenericSubarray[T]] = gops.wrap,
    ) -> Tuple[larr.FlatArray[T], Shape, Optional[MultiLabels]]:
        """
        Return a 'standard' `BlockArray` input format from general formats

        Parameters
        ----------
        subarrays, shape, labels :
            See class docstring

        Returns
        -------
        flat_subarrays, shape, labels
        """
        # Get the flat list of subarrays and the shape to validate the shape
        if isinstance(subarrays, larr.LabelledArray):
            flat_subarrays = subarrays.array.reshape(-1)
            implicit_shape = subarrays.f_shape
            implicit_labels = subarrays.f_labels
        elif isinstance(subarrays, (list, tuple)):
            flat_subarrays, implicit_shape = larr.flatten_array(subarrays)
            implicit_labels = None
        elif isinstance(subarrays, np.ndarray):
            implicit_shape = subarrays.shape
            flat_subarrays = subarrays.reshape(-1)
            implicit_labels = None
        else:
            raise TypeError(
                "Expected `subarrays` to be of type"
                " `{LabelledArray, list, tuple, np.ndarray}`"
                f" not {type(subarrays)}."
            )

        # If an explicit shape is not provided, assume the shape of `subarrays`
        # is the desired shape
        if shape is None:
            shape = implicit_shape
        if labels is None:
            labels = implicit_labels

        flat_subarrays = wrap(flat_subarrays)
        _validate_shape(flat_subarrays[0].ndim, shape)
        return flat_subarrays, shape, labels

    ## String representation functions
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.blocks)}, {self.f_shape}, {self.f_labels})"

    def __str__(self):
        return (
            f"{self.__class__.__name__}(bshape={self.f_bshape} labels={self.f_labels})"
        )

    ## Basic string representation functions
    def stats(self, stats=(np.min, np.max, np.mean)):
        """
        Return a dictionary of summary statistics for each subarray
        """
        multi_label_to_idx = self.larray._MULTI_LABEL_TO_IDX
        if any([axis_label_to_idx == () for axis_label_to_idx in multi_label_to_idx]):
            raise ValueError("Can't print state due to missing axis labels")

        dim_labels = [x.keys() for x in multi_label_to_idx]
        dim_idxs = [x.values() for x in multi_label_to_idx]
        return {
            idx_labels: tuple(stat(self[idx_ints]) for stat in stats)
            for idx_labels, idx_ints in zip(product(*dim_labels), product(*dim_idxs))
        }

    def print_summary(self):
        """
        Pretty-print the stats
        """
        import pprint as pp

        summary_dict = self.stats((np.min, np.max, np.mean))
        print('(min/max/mean):')
        pp.pprint(summary_dict)

    @property
    def blocks(self) -> np.ndarray:
        """
        Return the numpy object array containing subarrays
        """
        return self.larray.array

    @property
    def sub_blocks(self) -> np.ndarray:
        """
        Return the numpy object array containing subarrays
        """
        return gops.unwrap(self.larray.array)

    @property
    def larray(self) -> larr.LabelledArray:
        """
        Return the underlying labelled array
        """
        return self._larray

    @property
    def size(self) -> int:
        """
        Return the size (total number of blocks/subarrays)
        """
        return self.larray.size

    @property
    def f_shape(self) -> Shape:
        """
        Return the shape

        This is the number of blocks along each axis. The number of blocks along
        a reduced axis is returned as -1.
        """
        return self.larray.f_shape

    @property
    def f_bshape(self) -> BlockShape:
        """
        Return the block shape (shape of each block as a tuple)
        """
        return self._bshape

    @property
    def f_labels(self) -> MultiLabels:
        """Return the block labels"""
        return self.larray.f_labels

    @property
    def f_ndim(self):
        """
        Return the number of dimensions

        This is the number of axes in the block array, which matches the number
        of axes in all underlying sub-arrays.
        """
        return self.larray.f_ndim

    @property
    def f_dims(self):
        """
        Return a tuple of dimension indices

        This is simply a range from 0 to the number of dimensions.
        """
        return self.larray.f_dims

    @property
    def shape(self) -> Shape:
        """
        Return the reduced shape

        This is the number of blocks along each non-reduced axis.
        """
        return self.larray.shape

    @property
    def bshape(self) -> BlockShape:
        """
        Return the reduced block shape (number of blocks in each non-reduced axis)
        """
        ret_rbshape = [
            axis_sizes for axis_sizes in self.f_bshape if isinstance(axis_sizes, tuple)
        ]
        return tuple(ret_rbshape)

    @property
    def labels(self) -> MultiLabels:
        """
        Return the reduced block labels
        """
        return self._larray.labels

    @property
    def ndim(self):
        """
        Return the number of non-reduced dimensions

        This is the number of non-reduced axes.
        """
        return self.larray.ndim

    @property
    def dims(self):
        """
        Return a tuple of non-reduced dimension indices

        This is simply a tuple of the non-reduced dimensions.
        """
        return self.larray.dims

    @property
    def mshape(self) -> Shape:
        """
        Return the shape of the equivalent monolithic array
        """
        return tuple(axis_size(asize) for asize in self.f_bshape)

    ## Methods for converting to monolithic array
    def to_mono_ndarray(self):
        """
        Return a monolithic ndarray from the block array
        """
        return to_mono_ndarray(self)

    ## Copy methods
    def copy(self):
        """Return a copy"""
        labels = self.f_labels
        # Set `check_bshape=False` since the bshape was already checked
        return self.__class__(self.larray.copy(), labels=labels, check_bshape=False)

    def __copy__(self):
        return self.copy()

    @property
    def sub(self):
        """
        Return an object that allows indexing into unwrapped subarrays
        """

        class SubIndex:
            """
            Object to allow indexing into unwrapped subarrays
            """

            def __init__(self, barray: BlockArray):
                self._barray = barray

            def __getitem__(self, key: MultiGenIndex):
                result = self._barray[key]
                if isinstance(result, BlockArray):
                    result = result.blocks
                return gops.unwrap(result)

        return SubIndex(self)

    def __getitem__(self, key: MultiGenIndex) -> Union['BlockArray[T]', T]:
        """
        Return the vector or BlockVector corresponding to the index

        Parameters
        ----------
        key : str, int, slice
            A block label
        """
        ret = self.larray[key]
        if isinstance(ret, larr.LabelledArray):
            # Set `check_bshape=False` since the bshape was already checked
            # As long as `self.larray[key]` returns a propertly shaped selection
            # the bshape should already be valid
            return self.__class__(ret, check_bshape=False)
        else:
            return ret

    def __setitem__(self, key: MultiGenIndex, value: Union['BlockArray', T]):
        """
        Set subarrays to a given value

        Parameters
        ----------
        key :
            A general multi-index
        value :
            The desired value to set
        """
        set_array = self[key]
        if isinstance(set_array, BlockArray):
            # Set values to a `BlockArray` from a `BlockArray`
            if isinstance(value, BlockArray):
                if value.bshape != set_array.bshape:
                    raise ValueError(
                        f"Can't assign input values with bshape {value.bshape} to array with bshape {set_array.bshape}"
                    )
                for subarray, sub_value in zip(set_array, value.sub_blocks):
                    subarray.set(sub_value)
            # Set values to a flat `BlockArray` from a flat container
            elif isinstance(value, (list, tuple)):
                # Only allow assigning from flat lists to flat `set_array`
                if set_array.ndim != 1:
                    raise ValueError(
                        f"Can't assign list of input values to `BlockArray` with ndim {set_array.ndim}"
                    )
                elif set_array.size != len(value):
                    raise ValueError(
                        f"Can't assign list with {len(value)} items to `BlockArray` with {len(set_array)} blocks"
                    )
                else:
                    for subarray, subvalue in zip(set_array.blocks.flat, value):
                        subarray.set(subvalue)
            # Set values to a `BlockArray` from a single number
            elif isinstance(value, Number):
                for subarray in set_array.blocks.flat:
                    subarray.set(value)
            else:
                raise TypeError(f"Can't assign value of type {type(value)}")
        else:
            set_array.set(value)

    ## Reshape type methods
    def squeeze(self, axes=None):
        """
        Collapse axes blocks of size 1
        """
        if axes is None:
            axes = [ii for ii, size in enumerate(self.shape) if size == 1]
        f_axes = [self.dims[ii] for ii in axes]

        new_flabels = list(self.f_labels)
        new_fshape = list(self.f_shape)
        for ax in f_axes:
            if self.f_shape[ax] != 1:
                raise ValueError(f"Can't squeeze axis {ax:d} for shape {self.f_shape}")
            else:
                new_fshape[ax] = -1
                new_flabels[ax] = ()
        new_fshape = tuple(new_fshape)
        new_flabels = tuple(new_flabels)

        # NOTE: Squeezing probably shouldn't make sense for `BlockMatrix` and
        # `BlockVector` types
        return BlockArray(self.blocks.reshape(-1), new_fshape, new_flabels)

    def unsqueeze(self, f_axes=None):
        """
        Uncollapse axes blocks to size 1
        """
        if f_axes is None:
            f_axes = [ii for ii, size in enumerate(self.f_shape) if size == -1]

        new_fshape = unsqueeze_shape(self.f_shape, f_axes)
        # Unsqueezing `f_labels` doesn't require any modification
        new_flabels = self.f_labels

        return BlockArray(self.blocks.reshape(-1), new_fshape, new_flabels)

    ## Dict-like interface over the first dimension
    def keys(self):
        """Return the first axis' labels"""
        return self.larray.f_labels[0]

    def __contains__(self, key):
        return key in self.larray

    def items(self):
        """
        Return an iterable of label, value pairs along the first axis
        """
        return zip(self.f_labels[0], self)

    def sub_items(self):
        """
        Return an iterable of label, value (nonwrapped) pairs along the first axis
        """
        return zip(self.f_labels[0], self.sub_blocks)

    ## Iterable interface over the first non-reduced axis
    def __iter__(self):
        for ii in range(self.shape[0]):
            yield self[ii]

    def __sub_iter__(self):
        for ii in range(self.shape[0]):
            yield self.sub[ii]

    ## common operator overloading
    def __eq__(self, other):
        raise NotImplementedError()

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return scalar_mul(other, self)
        else:
            return mul(self, other)

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return scalar_div(other, self)
        else:
            return div(self, other)

    def __neg__(self):
        return neg(self)

    def __pos__(self):
        return pos(self)

    def __radd__(self, other):
        return add(other, self)

    def __rsub__(self, other):
        return sub(other, self)

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return scalar_mul(other, self)
        else:
            return mul(other, self)

    def __rtruediv__(self, other):
        if isinstance(other, (float, int)):
            return scalar_div(other, self)
        else:
            return div(other, self)

    ## Numpy ufunc interface
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Import this here to avoid ciruclar import errors
        # TODO: Fix bad module layout?
        from . import ufunc as _ufunc

        return _ufunc.apply_ufunc_array(ufunc, method, *inputs, **kwargs)


def _f_bshape_from_larray(larray: larr.LabelledArray[T]) -> BlockShape:
    """
    Return the block shape from subarrays in a `LabelledArray`

    The block shape is based on the subtensors along the 'boundary'. For an
    array of subtensors with shape `(2, 3, 4)` ...
    """
    blocks = larray.array
    f_ndim = larray.f_ndim
    dims = larray.dims
    _midx = [0] * f_ndim

    ret_bshape = []
    for dim, ax_size in enumerate(larray.f_shape):
        # If there are no blocks along a dimension,
        # the block axis size is an int
        if ax_size <= 0:
            axis_sizes = larray.array.flat[0].shape[dim]

            ret_bshape.append(axis_sizes)
        # If there are >= 1 blocks along a dimension, the block axis size is a
        # tuple of ints for each block along that dim
        else:
            midx = replace(_midx, [dim], [slice(None)])
            midx = tuple(midx[ii] for ii in dims)
            # Directly access subarrays in `.array` to avoid slow `LabelledArray.__getitem__`
            axis_sizes = tuple(subarray.shape[dim] for subarray in blocks[midx])

            ret_bshape.append(axis_sizes)

    return tuple(ret_bshape)


def _validate_f_bshape_from_larray(larray: larr.LabelledArray[T], f_bshape: BlockShape):
    """
    Validate subarrays have consistent shapes with supplied `f_bshape`

    The shapes of sub-arrays in a block format have to behave like a
    multiplication table to be consistent, where the 'edges' of the
    multiplication table are given by the block shape.

    Parameters
    ----------
    array : larr.LabelledArray
        The block array containing the subtensors
    f_bshape : BlockShape
        The 'full' block shape to validate
    """
    # Check that `larray.f_shape` and `f_bshape` agree
    assert len(larray.f_shape) == len(f_bshape)
    _f_shape = tuple(
        -1 if isinstance(bax_size, int) else len(bax_size) for bax_size in f_bshape
    )
    assert larray.f_shape == _f_shape

    # Check that subarray shapes agree with `f_bshape`
    # To validate subarray shapes, iterate through each block and check shapes
    # satisfy a multiplication table type rule:
    # `subarray[i, j, k, ...].shape == (bshape[i], bshape[j], bshape[k], ...)`
    # where `bshape` has any collapsed axes removed (this works because
    # `subarray[i, j, k, ...]` selects from non-collapsed axes).
    dims = tuple(ii for ii, bsize in enumerate(f_bshape) if not isinstance(bsize, int))
    bshape = tuple(f_bshape[ii] for ii in dims)
    shape = tuple(len(bsize) for bsize in bshape)
    midxs = [range(size) for size in shape]

    ref_subarr_shape = list(f_bshape)
    for midx, ref_shape in zip(product(*midxs), product(*bshape)):
        # Directly access subarrays in `.array` to avoid slow `LabelledArray.__getitem__`
        subarray_shape = larray.array[midx].shape
        # Update the axis size along non-collapsed axes
        ref_subarr_shape = replace(ref_subarr_shape, dims, ref_shape)
        if subarray_shape != tuple(ref_subarr_shape):
            raise ValueError(
                f"Subarray at {midx} with shape {subarray_shape} is inconsistent"
                f" with `f_bshape` {f_bshape}"
            )


def _validate_shape(ndim: int, shape: Shape):
    if len(shape) != ndim:
        raise ValueError(
            f"`shape` {shape} must have same number of dimensions as {ndim:d}"
        )


def axis_size(size: AxisSize) -> int:
    """
    Return the equivalent monolithic axis size from a block axis size
    """
    if isinstance(size, int):
        return size
    elif isinstance(size, tuple):
        return sum([axis_size(sub_size) for sub_size in size])
    else:
        raise TypeError(f"`size` must be int or tuple, not {type(size)}")


def axis_bsize(size: AxisSize) -> int:
    """
    Return the axis block size (number of blocks) from a block axis size
    """
    if isinstance(size, int):
        return -1
    elif isinstance(size, tuple):
        return len(size)
    else:
        raise TypeError(f"`size` must be int or tuple, not {type(size)}")


def unsqueeze_shape(shape: Shape, axes: Tuple[int, ...] = None) -> Shape:
    """
    Return a shape tuple with unreduced axes

    Parameters
    ----------
    shape :
        A shape tuple with potentially reduced axes
    axes :
        A tuple of int corresponding to which should be unreduced

    Returns
    -------
    ret_shape :
        The shape tuple with size 1 axes replacing previous size -1 axes.
    """

    if axes is None:
        axes = [ii for ii, size in enumerate(shape) if size == -1]

    ret_shape = list(shape)
    for ax in axes:
        if shape[ax] != -1:
            raise ValueError(f"Can't unsqueeze axis {ax:d} of shape {shape}")
        else:
            ret_shape[ax] = 1
    return tuple(ret_shape)


## `BlockArray` creation routines
def _require_tuple(ax_bsize: AxisSize) -> AxisSize:
    """
    Return non-tuple block axis sizes in a tuple

    This is similar to 'unsqueezing' any collapsed axes
    """
    if isinstance(ax_bsize, tuple):
        return ax_bsize
    elif isinstance(ax_bsize, int):
        return (ax_bsize,)
    else:
        raise TypeError(f"`ax_bshape` must be `tuple` or `int`, not {type(ax_bsize)}")


def make_create_array(create_numpy_array):
    """
    Derive a `BlockArray` creation routine from a `numpy` creation routine

    Parameters
    ----------
    create_numpy_array :
        A numpy array creation routine with the signature
        `create_numpy_array(shape, *args, **kwargs)`
        Examples are `np.zeros`, `np.ones`, etc.
    """

    def create_subarray(sub_shape):
        if all(isinstance(axsize, int) for axsize in sub_shape):
            return create_numpy_array(sub_shape)
        else:
            return create_block_array(sub_shape)

    def create_block_array(bshape):
        shape = tuple(axis_bsize(ax_bshape) for ax_bshape in bshape)

        _bshape = tuple(_require_tuple(ax_bshape) for ax_bshape in bshape)

        subarrays = [create_subarray(sub_shape) for sub_shape in product(*_bshape)]

        return BlockArray(subarrays, shape)

    return create_block_array


zeros = make_create_array(np.zeros)

ones = make_create_array(np.ones)

rand = make_create_array(lambda shape: np.random.rand(*shape))


## Binary operations
def _validate_elementwise_binary_op(a: BlockArray[T], b: BlockArray[T]):
    """
    Validates if BlockArray inputs are applicable
    """
    assert a.f_bshape == b.f_bshape


def _elementwise_binary_op(
    op: Callable[[T, T], T], a: BlockArray[T], b: BlockArray[T]
) -> BlockArray[T]:
    """
    Compute elementwise binary operation on block arrays

    This creates a new `BlockArray` with the same shape as the inputs by calling
    `op` on each pair of sub-arrays from the two inputs to create a
    corresponding output sub-array.

    Parameters
    ----------
    op: function
        A function with signature func(a, b) -> c, where a, b, c are vector of
        the same shape
    a, b: BlockArray
    """
    _validate_elementwise_binary_op(a, b)
    array = tuple(op(ai, bi) for ai, bi in zip(a.sub_blocks.flat, b.sub_blocks.flat))
    larrayay = larr.LabelledArray(array, a.f_shape, a.f_labels)
    return type(a)(larrayay)


add = functools.partial(_elementwise_binary_op, operator.add)

sub = functools.partial(_elementwise_binary_op, operator.sub)

mul = functools.partial(_elementwise_binary_op, operator.mul)

div = functools.partial(_elementwise_binary_op, operator.truediv)

power = functools.partial(_elementwise_binary_op, operator.pow)


## Unary operations
def _elementwise_unary_op(op: Callable[[T], T], a: BlockArray[T]) -> BlockArray[T]:
    """
    Compute elementwise unary operation on a BlockArray

    This creates a new `BlockArray` with the same shape by calling `op` on each
    sub-array.

    Parameters
    ----------
    op : Callable[[T], T]
        The operation to apply on the block array
    a : BlockArray
        The block array to apply the operation on

    Returns
    -------
    BlockArray
        The resultant block array
    """
    array = larr.LabelledArray(
        [op(ai) for ai in a.sub_blocks.flat], a.f_shape, a.f_labels
    )
    return type(a)(array)


neg = functools.partial(_elementwise_unary_op, operator.neg)

pos = functools.partial(_elementwise_unary_op, operator.pos)


def scalar_mul(alpha: Scalar, a: BlockArray[T]) -> BlockArray[T]:
    """
    Multiply a block array by a scalar
    """
    alpha = float(alpha)
    return _elementwise_unary_op(lambda subvec: alpha * subvec, a)


def scalar_div(alpha: Scalar, a: BlockArray[T]) -> BlockArray[T]:
    """
    Divide a block array by a scalar
    """
    alpha = float(alpha)
    return _elementwise_unary_op(lambda subvec: subvec / alpha, a)


def to_mono_ndarray(barray: BlockArray[T]) -> np.ndarray:
    """
    Convert a `BlockArray` to an equivalent monolithic `np.ndarray`

    Parameters
    ----------
    barray : BlockArray
        A block array object

    Returns
    -------
    np.ndarray
        A monolithic ndarray representing the input block array
    """
    # Get the shape of the monolithic array
    ret_array = np.zeros(barray.mshape)

    # cumulative block shape gives lower/upper block index bounds for assigning
    # individual blocks into the ndarray
    cum_r_bshape = [
        [nn for nn in accumulate(axis_shape, initial=0)] for axis_shape in barray.bshape
    ]

    # loop through each block and assign its elements to the appropriate
    # part of the monolithic ndarray
    for block_idx in product(*[range(n) for n in barray.shape]):
        lbs = [ax_strides[ii] for ii, ax_strides in zip(block_idx, cum_r_bshape)]
        ubs = [ax_strides[ii + 1] for ii, ax_strides in zip(block_idx, cum_r_bshape)]

        idxs = tuple(slice(lb, ub) for lb, ub in zip(lbs, ubs))

        midx = [slice(None)] * len(barray.f_shape)
        for ii, idx in zip(barray.dims, idxs):
            midx[ii] = idx
        ret_array[tuple(midx)] = barray.sub_blocks[block_idx]

    return ret_array
