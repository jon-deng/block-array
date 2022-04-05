"""
This module contains the block tensor definition which provides some basic operations
"""
from typing import TypeVar, Generic, Optional, Union, Callable
from itertools import accumulate
import functools

from . import vec as bvec
from . import array as barr
from . import subops as gops
# from . import blockmath as bmath

T = TypeVar('T')

def _block_shape(array):
    """
    Return the block shape of an array of subtensors
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

def validate_subtensor_shapes(array: barr.LabelledArray, bshape):
    """
    Validate subtensors in a BlockTensor have a valid shape

    array :
        The block array containing the subtensors
    shape :
        The target block shape of the BlockTensor
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
                ascblock_subtensors = array[ascblock_idx].array_flat
                ascblock_sizes = [
                    gops.shape(subtensor)[idx_ax] for subtensor in ascblock_subtensors]

                # compute the block sizes of all the remaining dimensions
                valid_bsizes = [bsize == _bsize for _bsize in ascblock_sizes]
                assert all(valid_bsizes)

class BlockTensor:
    """
    Represents a block vector with blocks indexed by keys

    Parameters
    ----------
    array :
        The subtensor elements
    shape :
        The shape of block tensor. For example, (2, 3) is a matrix block with
        2 row blocks by 3 columns blocks. A shape must be provided if `array` is
        a flat array.
    labels :
        Labels for each block along each axis
    """
    def __init__(
        self,
        array: Union[barr.LabelledArray, barr.NestedArray, barr.FlatArray],
        shape: Optional[barr.Shape] = None,
        labels: Optional[barr.AxisBlockLabels] = None):

        if isinstance(array, barr.LabelledArray):
            self._barray = array
        else:
            flat_array, _shape = barr.flatten_array(array)
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

            self._barray = barr.LabelledArray(flat_array, shape, labels)

        self._bshape = _block_shape(self._barray)

        validate_subtensor_shapes(self._barray, self.red_bshape)

    @property
    def subtensors_flat(self):
        """
        Return the flat tuple storing all subtensors
        """
        return self._barray.array_flat

    @property
    def subtensors_nested(self):
        """
        Return the nested tuple storing all subtensors
        """
        return self._barray.array_nested

    @property
    def barray(self):
        """
        Return the block array
        """
        return self._barray

    @property
    def labels(self):
        """Return the axis labels"""
        return self.barray.labels

    @property
    def size(self):
        """
        Return the size (total number of blocks)
        """
        return self.barray.size

    @property
    def shape(self):
        """
        Return the shape (number of blocks in each axis)
        """
        return self.barray.shape

    @property
    def rshape(self):
        """
        Return the reduced shape (number of blocks in each axis)
        """
        return self.barray.rshape

    @property
    def ndim(self):
        return self.barray.ndim

    @property
    def mshape(self):
        """
        Return the shape of the equivalent monolithic tensor
        """
        return tuple([sum(axis_sizes) for axis_sizes in self.bshape])

    @property
    def bshape(self):
        """
        Return the block shape (shape of each block as a tuple)
        """
        return self._bshape

    @property
    def red_bshape(self):
        """
        Return the reduced block shape (number of blocks in each axis)
        """
        ret_rbshape = [axis_sizes for axis_sizes in self.bshape if axis_sizes != ()]
        return ret_rbshape

    @property
    def red_mshape(self):
        """
        Return the reduced block size (number of blocks in each axis)
        """
        ret_rbsize = [axis_size for axis_size in self.mshape if axis_size != 0]
        return ret_rbsize

    ## Copy methods
    def copy(self):
        """Return a copy"""
        labels = self.labels
        return self.__class__(self.barray.copy(), labels)

    def __copy__(self):
        return self.copy()

    def __getitem__(self, key):
        """
        Return the vector or BlockVec corresponding to the index

        Parameters
        ----------
        key : str, int, slice
            A block label
        """
        ret = self.barray[key]
        if isinstance(ret, barr.LabelledArray):
            return self.__class__(ret)
        else:
            return ret

    ## Dict-like interface over the first dimension
    @property
    def keys(self):
        """Return the first axis' labels"""
        return self.barray.labels[0]

    def __contains__(self, key):
        return key in self.barray

    def items(self):
        return zip(self.labels[0], self)

    ## Iterable interface over the first non-reduced axis
    def __iter__(self):
        for ii in range(self.rshape[0]):
            yield self[ii]

    ## common operator overloading
    def __eq__(self, other):
        eq = False
        if isinstance(other, BlockVec):
            err = self - other
            if dot(err, err) == 0:
                eq = True
        else:
            raise TypeError(f"Cannot compare {type(other)} to {type(self)}")

        return eq

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

def validate_elementwise_binary_op(a, b):
    """
    Validates if BlockTensor inputs are applicable
    """
    assert a.bshape == b.bshape

def _elementwise_binary_op(op: Callable[T, T], a: BlockTensor, b: BlockTensor):
    """
    Compute elementwise binary operation on BlockTensors

    Parameters
    ----------
    op: function
        A function with signature func(a, b) -> c, where a, b, c are vector of
        the same shape
    a, b: BlockTensor
    """
    validate_elementwise_binary_op(a, b)
    array = tuple([op(ai, bi) for ai, bi in zip(a.subtensors_flat, b.subtensors_flat)])
    barray = barr.LabelledArray(array, a.shape, a.labels)
    return type(a)(barray)

add = functools.partial(_elementwise_binary_op, lambda a, b: a+b)

sub = functools.partial(_elementwise_binary_op, lambda a, b: a-b)

mul = functools.partial(_elementwise_binary_op, lambda a, b: a*b)

div = functools.partial(_elementwise_binary_op, lambda a, b: a/b)

power = functools.partial(_elementwise_binary_op, lambda a, b: a**b)


def _elementwise_unary_op(op: Callable, a: BlockTensor):
    """
    Compute elementwise unary operation on a BlockTensor

    Parameters
    ----------
    a: BlockTensor
    """
    array = barr.LabelledArray([op(ai) for ai in a.subtensors_flat], a.shape, a.labels)
    return type(a)(array)

neg = functools.partial(_elementwise_unary_op, lambda a: -a)

pos = functools.partial(_elementwise_unary_op, lambda a: +a)

def scalar_mul(alpha, a):
    return _elementwise_unary_op(lambda subvec: alpha*subvec, a)

def scalar_div(alpha, a):
    return _elementwise_unary_op(lambda subvec: subvec/alpha, a)

def to_ndarray(block_tensor: BlockTensor):
    """
    Convert a BlockTensor object to a ndarray object
    """
    # .bsize (block size) is the resulting shape of the monolithic array
    ret_array = np.zeros(block_tensor.mshape)

    # cumulative block shape gives lower/upper block index bounds for assigning
    # individual blocks into the ndarray
    cum_bshape = [
        [nn for nn in accumulate(axis_shape, initial=0)]
        for axis_shape in block_tensor.bshape]

    # loop through each block and assign its elements to the appropriate
    # part of the monolithic ndarray
    for block_idx in product(*[range(axis_size) for axis_size in block_tensor.shape]):
        lbs = [cum_bshape[ii] for ii in block_idx]
        ubs = [cum_bshape[ii+1] for ii in block_idx]

        idx = tuple([slice(lb, ub) for lb, ub in zip(lbs, ubs)])
        ret_array[idx] = block_tensor[block_idx]

    return ret_array
