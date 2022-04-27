"""
This module contains the block tensor definition which provides some basic operations
"""
from typing import TypeVar, Optional, Union, Callable
import itertools
import functools
import operator
import numpy as np

from . import labelledarray as larr
from . import subops as gops
from .typing import (BlockShape, Shape, MultiLabels, Scalar)

T = TypeVar('T')

def _block_shape(array: larr.LabelledArray) -> BlockShape:
    """
    Return the block shape of an array of subtensors

    The block shape is based on the subtensors along the 'boundary'. For an
    array of subtensors with shape `(2, 3, 4)` ...
    """

    ret_bshape = []
    num_axes = len(array.shape)
    for idx_ax, num_blocks in enumerate(array.shape):
        axis_sizes = []
        for idx_block in range(num_blocks):
            idx = tuple((idx_ax)*[0] + [idx_block] + (num_axes-idx_ax-1)*[0])
            # remove indices along reduced dimensions
            ridx = tuple([
                idx_ax for nax, idx_ax in enumerate(idx)
                if array.shape[nax] != -1])

            block_axis_size = gops.shape(array[ridx])[idx_ax]
            axis_sizes.append(block_axis_size)
        ret_bshape.append(tuple(axis_sizes))
    return tuple(ret_bshape)

def validate_subtensor_shapes(array: larr.LabelledArray, bshape: BlockShape):
    """
    Validate subtensors in a BlockArray have a valid shape

    array :
        The block array containing the subtensors
    shape :
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

class BlockArray:
    """
    An n-dimensional block tensor object

    `BlockArray` has two main attributes: an underlying `LabelledArray` that stores 
    the subtensors in an n-d layout and a `bshape` attributes that stores the shape 
    of the blocks along each axis.

    For example, consider a `BlockArray` `A` with the block shape `((10, 5), (2, 4))`.
    This represents a matrix with blocks of size 10 and 5 along the rows, and blocks 
    of size 2 and 4 along the columns. Subtensors of `A` would then have shapes:
        `A[0, 0].shape == (10, 2)`
        `A[0, 1].shape == (10, 4)`
        `A[1, 0].shape == (5, 2)`
        `A[1, 1].shape == (5, 4)`

    Parameters
    ----------
    array :
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
    larray : larr.LabelledArray
        The `LabelledArray` instance used to store the subtensors in a block format

    size :
        The total number of subarrays contained in the block array
    shape :
        The shape of the block array
    bshape :
        The block shape of the block array. A nested tuple representing the sizes
        of each block along each axis.
    mshape :
        The monolithic shape of the block array
    r_shape, r_bshape, r_mshape :
        Reduced version of `shape`, `bshape` and `mshape`, respectively. These
        have the same format as their correponding attributes but do not
        include any reduced/collapsed dimensions.
    ndim :
        The number of dimensions
    r_ndim :
        The number of reduced dimensions
    dims :
        A tuples of indices for each dimension
    r_dims :
        A tuple of indices for each non-reduced dimension

    subarrays_flat : 
        A flat tuple of the contained subarrays
    subarrays_nest :
        A nested tuple of the contained subarrays
    """
    def __init__(
        self,
        array: Union[larr.LabelledArray, larr.NestedArray, larr.FlatArray],
        shape: Optional[Shape] = None,
        labels: Optional[MultiLabels] = None):

        if isinstance(array, larr.LabelledArray):
            self._larray = array
        else:
            flat_array, _shape = larr.flatten_array(array)
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

        self._bshape = _block_shape(self._larray)

        validate_subtensor_shapes(self._larray, self.r_bshape)

    ## String representation functions
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.subarrays_flat)}, {self.shape}, {self.labels})"

    def __str__(self):
        return f"{self.__class__.__name__}(bshape={self.bshape} labels={self.labels})"

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
    def labels(self):
        """Return the axis labels"""
        return self.larray.labels

    @property
    def size(self):
        """
        Return the size (total number of blocks)
        """
        return self.larray.size

    @property
    def shape(self):
        """
        Return the shape (number of blocks in each axis)
        """
        return self.larray.shape

    @property
    def r_shape(self):
        """
        Return the reduced shape (number of blocks in each axis)
        """
        return self.larray.r_shape

    @property
    def ndim(self):
        return self.larray.ndim

    @property
    def r_ndim(self):
        return self.larray.r_ndim

    @property
    def dims(self):
        return self.larray.dims

    @property
    def r_dims(self):
        return self.larray.r_dims

    @property
    def mshape(self) -> Shape:
        """
        Return the shape of the equivalent monolithic tensor
        """
        return tuple([sum(axis_sizes) for axis_sizes in self.bshape])

    @property
    def bshape(self) -> BlockShape:
        """
        Return the block shape (shape of each block as a tuple)
        """
        return self._bshape

    @property
    def r_bshape(self) -> BlockShape:
        """
        Return the reduced block shape (number of blocks in each axis)
        """
        ret_rbshape = [axis_sizes for axis_sizes in self.bshape if axis_sizes != ()]
        return ret_rbshape

    @property
    def r_mshape(self) -> Shape:
        """
        Return the reduced block size (number of blocks in each axis)
        """
        ret_rbsize = [axis_size for axis_size in self.mshape if axis_size != 0]
        return ret_rbsize

    ## Copy methods
    def copy(self):
        """Return a copy"""
        labels = self.labels
        return self.__class__(self.larray.copy(), labels)

    def __copy__(self):
        return self.copy()

    def __getitem__(self, key):
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
    @property
    def keys(self):
        """Return the first axis' labels"""
        return self.larray.labels[0]

    def __contains__(self, key):
        return key in self.larray

    def items(self):
        return zip(self.labels[0], self)

    ## Iterable interface over the first non-reduced axis
    def __iter__(self):
        for ii in range(self.r_shape[0]):
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
    def __array_ufunc__(ufunc, method, *inputs, **kwargs):
        for btensor in inputs:
            if not isinstance(btensor, BlockArray):
                raise TypeError(
                    f"ufunc {ufunc} cannot be called with inputs" +
                    ", ".join([f"{type(input)}" for input in inputs])
                    )

        subtensors_in = [btensor.subtensors_flat for btensor in inputs]
        subtensors_out = [
            ufunc(*subtensor_inputs) for subtensor_inputs in subtensors_in
            ]
        ret_shape = inputs[0].shape
        ret_labels = inputs[0].labels

        return BlockArray(subtensors_out, ret_shape, ret_labels)

def validate_elementwise_binary_op(a: BlockArray, b: BlockArray):
    """
    Validates if BlockArray inputs are applicable
    """
    assert a.bshape == b.bshape

def _elementwise_binary_op(
        op: Callable[[T, T], T], 
        a: BlockArray, b: BlockArray
    ) -> BlockArray:
    """
    Compute elementwise binary operation on BlockArrays

    Parameters
    ----------
    op: function
        A function with signature func(a, b) -> c, where a, b, c are vector of
        the same shape
    a, b: BlockArray
    """
    validate_elementwise_binary_op(a, b)
    array = tuple([op(ai, bi) for ai, bi in zip(a.subarrays_flat, b.subarrays_flat)])
    larrayay = larr.LabelledArray(array, a.shape, a.labels)
    return type(a)(larrayay)

add = functools.partial(_elementwise_binary_op, operator.add)

sub = functools.partial(_elementwise_binary_op, operator.sub)

mul = functools.partial(_elementwise_binary_op, operator.mul)

div = functools.partial(_elementwise_binary_op, operator.truediv)

power = functools.partial(_elementwise_binary_op, operator.pow)


def _elementwise_unary_op(
        op: Callable[[T], T], a: BlockArray
    ) -> BlockArray:
    """
    Compute elementwise unary operation on a BlockArray

    Parameters
    ----------
    a: BlockArray
    """
    array = larr.LabelledArray([op(ai) for ai in a.subarrays_flat], a.shape, a.labels)
    return type(a)(array)

neg = functools.partial(_elementwise_unary_op, operator.neg)

pos = functools.partial(_elementwise_unary_op, operator.pos)

def scalar_mul(alpha: Scalar, a: BlockArray) -> BlockArray:
    return _elementwise_unary_op(lambda subvec: alpha*subvec, a)

def scalar_div(alpha: Scalar, a: BlockArray) -> BlockArray:
    return _elementwise_unary_op(lambda subvec: subvec/alpha, a)

def to_ndarray(block_tensor: BlockArray):
    """
    Convert a BlockArray object to a ndarray object
    """
    # .bsize (block size) is the resulting shape of the monolithic array
    ret_array = np.zeros(block_tensor.mshape)

    # cumulative block shape gives lower/upper block index bounds for assigning
    # individual blocks into the ndarray
    cum_bshape = [
        [nn for nn in itertools.accumulate(axis_shape, initial=0)]
        for axis_shape in block_tensor.bshape]

    # loop through each block and assign its elements to the appropriate
    # part of the monolithic ndarray
    for block_idx in itertools.product(*[range(axis_size) for axis_size in block_tensor.shape]):
        lbs = [cum_bshape[ii] for ii in block_idx]
        ubs = [cum_bshape[ii+1] for ii in block_idx]

        idx = tuple([slice(lb, ub) for lb, ub in zip(lbs, ubs)])
        ret_array[idx] = block_tensor[block_idx]

    return ret_array
