"""
This module contains the block tensor definition which provides some basic operations
"""
from typing import TypeVar, Generic, Optional

from . import blockarray as barr
from . import genericops as gops

T = TypeVar('T')

class BlockTensor(Generic[T]):
    """
    Represents a block vector with blocks indexed by keys

    Parameters
    ----------
    subtensors : tuple(PETsc.Vec or dolfin.cpp.la.PETScVector or np.ndarray)
    keys : tuple(str)
    """
    def __init__(self, blockarray):
        self._array = blockarray

    @property
    def barray(self):
        return self._array

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