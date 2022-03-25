"""
This module contains the block tensor definition which provides some basic operations
"""
from typing import TypeVar, Generic, Optional, Union
from itertools import accumulate

from . import blockarray as barr
from . import genericops as gops

# T = TypeVar('T')

def to_ndarray(block_tensor: BlockTensor):
    """
    Convert a BlockTensor object to a ndarray object
    """
    # .bsize (block size) is the resulting shape of the monolithic array
    ret_array = np.zeros(block_tensor.bsize)

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

class BlockTensor:
    """
    Represents a block vector with blocks indexed by keys

    Parameters
    ----------
    subtensors : tuple(PETsc.Vec or dolfin.cpp.la.PETScVector or np.ndarray)
    keys : tuple(str)
    """
    def __init__(
        self, 
        barray: Union[barr.BlockArray, barr.NestedArray],
        labels: Optional[barr.AxisBlockLabels] = None):

        if isinstance(barray, barr.BlockArray):
            self._barray = barray
        else:
            self._barray = barr.block_array(barray, labels)

    @property
    def array(self):
        """
        Return the flat tuple storing all subtensors
        """
        return self._barray.array

    @property
    def array_nested(self):
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
    def ndim(self):
        return self.barray.ndim

    @property
    def bsize(self):
        """
        Return the block size (total size of each block for each axis)
        """
        return tuple([sum(axis_sizes) for axis_sizes in self.bshape])
        
    @property
    def bshape(self):
        """
        Return the block shape (shape of each block as a tuple)
        """
        ret_bshape = []
        num_axes = len(self.shape)
        for nax, num_blocks in enumerate(self.shape):
            axis_sizes = []
            for nblock in range(num_blocks):
                index = tuple((nax)*[0] + [nblock] + (num_axes-nax-1)*[0])
                axis_size = gops.shape(self.barray[index])[nax]
                axis_sizes.append(axis_size)
            ret_bshape.append(tuple(axis_sizes))
        return tuple(ret_bshape)

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
        if isinstance(ret, barr.BlockArray):
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

    ## Iterable interface over the first axis
    def __iter__(self):
        for ii in range(self.shape[0]):
            yield self[ii]
