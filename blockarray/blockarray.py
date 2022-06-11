"""
This module contains the block array definition and defines some basic operations
"""

from typing import TypeVar, Optional, Union, Callable, Generic
import itertools
import functools
import operator
import numpy as np

from . import labelledarray as larr
from . import subops as gops
from .typing import (BlockShape, Shape, MultiLabels, Scalar, MultiGenIndex)

T = TypeVar('T')

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
    shape :
        The number of blocks (subarrays) along each axis. For example, a matrix
        with 2 row blocks and 2 column blocks has shape `(2, 2)`.
    bshape :
        The 'block shape' (or nested shape) of the block array. This stores the
        axis sizes of subarrays along each axis. For example, a block shape
        `((120, 6), (5, 4))` represents a 2-by-2 block matrix with entries:
            - (0, 0) is a 120x5 matrix
            - (0, 1) is a 120x4 matrix
            - (1, 0) is a 6x5 matrix
            - (1, 1) is a 6x4 matrix
    mshape :
        The shape of the block array's monolithic equivalent.
    r_shape, r_bshape :
        Reduced version of `shape` and `bshape` respectively. These
        have the same format as their correponding attributes but do not
        include any reduced/collapsed dimensions.
    ndim :
        The number of dimensions
    r_ndim :
        The number of non-reduced/collapsed dimensions
    dims :
        A tuples of indices for each dimension
    r_dims :
        A tuple of indices for each non-reduced/collapsed dimension
    labels :
        A tuple of labels for each block along each axis

    subarrays_flat :
        A flat tuple of the contained subarrays
    subarrays_nest :
        A nested tuple of the contained subarrays

    larray : larr.LabelledArray
        The `LabelledArray` instance used to store the subtensors in a block
        format
    """
    def __init__(
            self,
            subarrays: Union[larr.LabelledArray[T], larr.NestedArray[T], larr.FlatArray[T]],
            shape: Optional[Shape]=None,
            labels: Optional[MultiLabels]=None
        ):

        if isinstance(subarrays, larr.LabelledArray):
            self._larray = subarrays
        elif isinstance(subarrays, (list, tuple)):
            flat_array, _shape = larr.flatten_array(subarrays)
            if shape is None:
                # If a shape is not provided, assume `array` is a nested
                # array representation
                shape = _shape
            elif len(_shape) > 1:
                # If a shape is provided for a nested array, check that nested
                # array shape and provided shape are compatible
                if shape != _shape:
                    raise ValueError(
                        "Nested array shape {_shape} and provided shape {shape}"
                        "are not compatible")
            self._larray = larr.LabelledArray(flat_array, shape, labels)
        else:
            raise TypeError(
                "Expected `subarrays` to be of type `LabelledArray`, `list`, or `tuple`"
                f" not {type(subarrays)}."
            )

        self._bshape = _block_shape_from_larray(self._larray)

        _validate_subarray_shapes_from_larray(self._larray, self.bshape)

    ## String representation functions
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.subarrays_flat)}, {self.f_shape}, {self.f_labels})"

    def __str__(self):
        return f"{self.__class__.__name__}(bshape={self.f_bshape} labels={self.f_labels})"

    @property
    def subarrays_flat(self):
        """
        Return the flat tuple storing all subtensors
        """
        return self._larray.flat

    @property
    def subarrays_nest(self):
        """
        Return the nested tuple storing all subtensors
        """
        return self._larray.nest

    @property
    def larray(self) -> larr.LabelledArray:
        """
        Return the underlying labelled array
        """
        return self._larray

    @property
    def size(self):
        """
        Return the size (total number of blocks/subarrays)
        """
        return self.larray.size

    @property
    def f_shape(self):
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
    def f_labels(self):
        """Return the axis labels"""
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
    def shape(self):
        """
        Return the reduced shape

        This is the number of block along each non-reduced axis.
        """
        return self.larray.shape

    @property
    def bshape(self) -> BlockShape:
        """
        Return the reduced block shape (number of blocks in each non-reduced axis)
        """
        ret_rbshape = [axis_sizes for axis_sizes in self.f_bshape if axis_sizes != ()]
        return tuple(ret_rbshape)

    @property
    def labels(self):
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
        _mshape = gops.shape(self.subarrays_flat[0])
        return tuple([max(sum(axis_sizes), r_size) for r_size, axis_sizes in zip(_mshape, self.f_bshape)])

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
        return self.__class__(self.larray.copy(), labels)

    def __copy__(self):
        return self.copy()

    def __getitem__(self, key: MultiGenIndex):
        """
        Return the vector or BlockVector corresponding to the index

        Parameters
        ----------
        key : str, int, slice
            A block label
        """
        ret = self.larray[key]
        if isinstance(ret, larr.LabelledArray):
            return self.__class__(ret)
        else:
            return ret

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

    ## Iterable interface over the first non-reduced axis
    def __iter__(self):
        for ii in range(self.shape[0]):
            yield self[ii]

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

def _block_shape_from_larray(array: larr.LabelledArray[T]) -> BlockShape:
    """
    Return the block shape from subarrays in a `LabelledArray`

    The block shape is based on the subtensors along the 'boundary'. For an
    array of subtensors with shape `(2, 3, 4)` ...
    """
    ret_bshape = []
    num_axes = len(array.f_shape)
    for idx_ax, num_blocks in enumerate(array.f_shape):
        axis_sizes = []
        for idx_block in range(num_blocks):
            idx = tuple((idx_ax)*[0] + [idx_block] + (num_axes-idx_ax-1)*[0])
            # remove indices along reduced dimensions
            ridx = tuple([
                idx_ax for nax, idx_ax in enumerate(idx)
                if array.f_shape[nax] != -1])

            block_axis_size = gops.shape(array[ridx])[idx_ax]
            axis_sizes.append(block_axis_size)
        ret_bshape.append(tuple(axis_sizes))
    return tuple(ret_bshape)

def _validate_subarray_shapes_from_larray(
        array: larr.LabelledArray[T],
        bshape: BlockShape
    ):
    """
    Validate sub-arrays in a `LabelledArray` have consistent shapes

    The shapes of sub-arrays in a block format have to behave like a
    multiplication table to be consistent, where the 'edges' of the
    multiplication table are given by the block shape.

    Parameters
    ----------
    array : larr.LabelledArray
        The block array containing the subtensors
    shape : BlockShape
        The target block shape of the BlockArray
    """
    # Subtensors are valid along an axis at a certain block if:
    # all subtensors in the remaining dimensions have the same shape along that
    # axis (i.e. shape[axis_idx] is the same for all subtensors in the remaining
    # dimensions)
    ndim = len(bshape)
    if ndim > 1:
        # no need to check compatibility for 1 dimensional tensors
        for idx_ax, bsizes in enumerate(bshape):
            for idx_block, bsize in enumerate(bsizes):
                # index all subtensors with the associated (asc) `idx_block`,
                # i.e. all subtensors along the remaining axes
                ascblock_idx = (
                    (slice(None),)*idx_ax
                    + (idx_block,)
                    + (slice(None),)*(ndim-idx_ax-1))
                ascblock_subtensors = array[ascblock_idx].flat
                ascblock_sizes = [
                    gops.shape(subtensor)[idx_ax] for subtensor in ascblock_subtensors]

                # compute the block sizes of all the remaining dimensions
                valid_bsizes = [bsize == _bsize for _bsize in ascblock_sizes]
                assert all(valid_bsizes)


def _validate_elementwise_binary_op(a: BlockArray[T], b: BlockArray[T]):
    """
    Validates if BlockArray inputs are applicable
    """
    assert a.f_bshape == b.f_bshape

def _elementwise_binary_op(
        op: Callable[[T, T], T],
        a: BlockArray[T],
        b: BlockArray[T]
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
    array = tuple([op(ai, bi) for ai, bi in zip(a.subarrays_flat, b.subarrays_flat)])
    larrayay = larr.LabelledArray(array, a.f_shape, a.f_labels)
    return type(a)(larrayay)

add = functools.partial(_elementwise_binary_op, operator.add)

sub = functools.partial(_elementwise_binary_op, operator.sub)

mul = functools.partial(_elementwise_binary_op, operator.mul)

div = functools.partial(_elementwise_binary_op, operator.truediv)

power = functools.partial(_elementwise_binary_op, operator.pow)


def _elementwise_unary_op(
        op: Callable[[T], T], a: BlockArray[T]
    ) -> BlockArray[T]:
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
    array = larr.LabelledArray([op(ai) for ai in a.subarrays_flat], a.f_shape, a.f_labels)
    return type(a)(array)

neg = functools.partial(_elementwise_unary_op, operator.neg)

pos = functools.partial(_elementwise_unary_op, operator.pos)

def scalar_mul(alpha: Scalar, a: BlockArray[T]) -> BlockArray[T]:
    """
    Multiply a block array by a scalar
    """
    return _elementwise_unary_op(lambda subvec: alpha*subvec, a)

def scalar_div(alpha: Scalar, a: BlockArray[T]) -> BlockArray[T]:
    """
    Divide a block array by a scalar
    """
    return _elementwise_unary_op(lambda subvec: subvec/alpha, a)

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
        [nn for nn in itertools.accumulate(axis_shape, initial=0)]
        for axis_shape in barray.bshape]

    # loop through each block and assign its elements to the appropriate
    # part of the monolithic ndarray
    for block_idx in itertools.product(*[range(axis_size) for axis_size in barray.shape]):
        lbs = [ax_strides[ii] for ii, ax_strides in zip(block_idx, cum_r_bshape)]
        ubs = [ax_strides[ii+1] for ii, ax_strides in zip(block_idx, cum_r_bshape)]

        idxs = tuple([slice(lb, ub) for lb, ub in zip(lbs, ubs)])

        midx = [slice(None)]*len(barray.f_shape)
        for ii, idx in zip(barray.dims, idxs):
            midx[ii] = idx
        ret_array[tuple(midx)] = barray[block_idx]

    return ret_array
