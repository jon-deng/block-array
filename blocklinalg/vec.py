"""
This module contains the block vector definition and various operations on 
block vectors
"""

from typing import TypeVar, Generic, List, Optional

import numpy as np
from petsc4py import PETSc

from . import genericops as gops
from . import blockarray as barr
from .tensor import BlockTensor

## pylint: disable=no-member

# TODO: Rename keys to labels

# Type variable for a 'sub'vector
T = TypeVar('T')

# BlockVec methods
def split_bvec(bvec, block_sizes):
    """
    Splits a block vector into multiple block vectors
    """
    split_bvecs = []
    _bvec = bvec
    for bsize in block_sizes:
        split_bvecs.append(_bvec[:bsize])
        _bvec = _bvec[bsize:]
    return tuple(split_bvecs)

def concatenate_vec(args):
    """
    Concatenate a series of BlockVecs into a single BlockVec

    Parameters
    ----------
    args : List of BlockVec
    """
    vecs = []
    keys = []
    for bvec in args:
        vecs += bvec.vecs
        keys += bvec.keys

    return BlockVec(vecs, keys)

def validate_blockvec_size(*args):
    """
    Check if a collection of BlockVecs have compatible block sizes
    """
    size = args[0].size
    for arg in args:
        if arg.size != size:
            return False

    return True

def handle_scalars(bvec_op):
    """
    Decorator to handle scalar inputs to BlockVec functions
    """
    def wrapped_bvec_op(*args):
        # Find all input BlockVec arguments and check if they have compatible sizes
        bvecs = [arg for arg in args if isinstance(arg, BlockVec)]
        if not validate_blockvec_size(*bvecs):
            raise ValueError(f"Could not perform operation on BlockVecs with sizes", [vec.size for vec in bvecs])

        bsize = bvecs[0].size
        keys = bvecs[0].keys

        # Convert floats to scalar BlockVecs in a new argument list
        new_args = []
        for arg in args:
            if isinstance(arg, float):
                _vecs = tuple([arg]*bsize)
                new_args.append(BlockVec(_vecs, keys))
            elif isinstance(arg, np.ndarray) and arg.shape == ():
                _vecs = tuple([float(arg)]*bsize)
                new_args.append(BlockVec(_vecs, keys))
            else:
                new_args.append(arg)

        return bvec_op(*new_args)
    return wrapped_bvec_op

def dot(a, b):
    """
    Return the dot product of a and b
    """
    c = a*b
    ret = 0
    for vec in c:
        # using the [:] indexing notation makes sum interpret the different data types as np arrays 
        # which can improve performance a lot
        ret += sum(vec[:])
    return ret

@handle_scalars
def add(a, b):
    """
    Add block vectors a and b

    Parameters
    ----------
    a, b: BlockVec or float
    """
    keys = a.keys
    vecs = tuple([ai+bi for ai, bi in zip(a.vecs, b.vecs)])
    return BlockVec(vecs, keys)

@handle_scalars
def sub(a, b):
    """
    Subtract block vectors a and b

    Parameters
    ----------
    a, b: BlockVec or float
    """
    keys = a.keys
    vecs = tuple([ai-bi for ai, bi in zip(a, b)])
    return BlockVec(vecs, keys)

@handle_scalars
def mul(a, b):
    """
    Elementwise multiplication of block vectors a and b

    Parameters
    ----------
    a, b: BlockVec or float
    """
    keys = a.keys
    vecs = tuple([ai*bi for ai, bi in zip(a.vecs, b.vecs)])
    return BlockVec(vecs, keys)

@handle_scalars
def div(a, b):
    """
    Elementwise division of block vectors a and b

    Parameters
    ----------
    a, b: BlockVec or float
    """
    keys = a.keys
    vecs = tuple([ai/bi for ai, bi in zip(a, b)])
    return BlockVec(vecs, keys)

@handle_scalars
def power(a, b):
    """
    Elementwise power of block vector a to b

    Parameters
    ----------
    a, b: BlockVec or float
    """
    keys = a.keys
    vecs = tuple([ai**bi for ai, bi in zip(a, b)])
    return BlockVec(vecs, keys)

@handle_scalars
def neg(a):
    """
    Negate block vector a

    Parameters
    ----------
    a: BlockVec
    """
    keys = a.keys
    vecs = tuple([-ai for ai in a])
    return BlockVec(vecs, keys)

@handle_scalars
def pos(a):
    """
    Positifiy block vector a
    """
    keys = a.keys
    vecs = tuple([+ai for ai in a])
    return BlockVec(vecs, keys)

def convert_bvec_to_petsc(bvec):
    """
    Converts a block matrix from one submatrix type to the PETSc submatrix type

    Parameters
    ----------
    bmat: BlockMat
    """
    vecs = [gops.convert_vec_to_petsc(subvec) for subvec in bvec.vecs]
    return BlockVec(vecs, bvec.keys)

class BlockVec(BlockTensor):
    """
    Represents a block vector with blocks indexed by keys

    Parameters
    ----------
    vecs : tuple(PETsc.Vec or dolfin.cpp.la.PETScVector or np.ndarray)
    keys : tuple(str)
    """

    @property
    def vecs(self):
        """Return tuple of vectors from each block"""
        return self.barray.array
    
    ## Basic string representation functions
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        desc = ", ".join([f"{key}:{gops.size_vec(vec)}" for key, vec in zip(self.keys, self.vecs)])
        return f"({desc})"

    ## Dict-like interface
    def __contains__(self, key):
        return key in self.barray

    def __iter__(self):
        for key in self.keys:
            yield self.barray[key]

    def items(self):
        return zip(self.keys, self.vecs)

    def print_summary(self):
        summary_strings = [
            f"{key}: ({np.min(vec[:])}/{np.max(vec[:])}/{np.mean(vec[:])})"
            for key, vec in self.items()]

        summary_message = ", ".join(["block: (min/max/mean)"] + summary_strings)
        print(summary_message)
        return summary_message

    ## Array/dictionary-like slicing and indexing interface
    @property
    def monovec(self):
        """
        Return an object allowing indexing of the block vector as a monolithic vector
        """
        return MonotoBlock(self)
    
    def __setitem__(self, key, value):
        """
        Return the vector corresponding to the labelled block

        Parameters
        ----------
        key : str, int, slice
            A block label
        value : array_like or BlockVec
        """
        _array = self.barray[key]
        if isinstance(_array, barr.BlockArray):
            for subvec in _array.array:
                gops.set_vec(subvec, value)
        else:
            gops.set_vec(_array, value)

    def set(self, scalar):
        """
        Set a constant value for the block vector
        """
        for vec in self:
            gops.set_vec(vec, scalar)

    def set_vec(self, vec):
        """
        Sets all values based on a vector
        """
        # Check sizes are compatible
        assert vec.size == np.sum(self.bsize)

        # indices of the boundaries of each block
        n_blocks = np.concatenate(([0], np.cumsum(self.bsize)))
        for i, (n_start, n_stop) in enumerate(zip(n_blocks[:-1], n_blocks[1:])):
            self[i][:] = vec[n_start:n_stop]

    ## Conversion and treatment as a monolithic vector
    def to_ndarray(self):
        ndarray_vecs = [np.array(vec) for vec in self.vecs]
        return np.concatenate(ndarray_vecs, axis=0)

    def to_petsc_seq(self, comm=None):
        total_size = np.sum(self.bsize)
        vec = PETSc.Vec.createSeq(total_size, comm=comm)
        vec.setArray(self.to_ndarray)
        vec.assemblyBegin()
        vec.assemblyEnd()
        return vec

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
        return mul(self, other)

    def __truediv__(self, other):
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
        return mul(other, self)

    def __rtruediv__(self, other):
        return div(other, self)

    ## 
    def norm(self):
        return dot(self, self)**0.5

class MonotoBlock:
    def __init__(self, bvec):
        self.bvec = bvec

    def __setitem__(self, key, value):
        assert isinstance(key, slice)
        total_size = np.sum(self.bvec.size)

        # Let n refer to the monolithic index while m refer to a block index (the # of blocks)
        nstart, nstop, nstep = self.slice_to_numeric(key, total_size)

        # Get the monolithic ending indices of each block
        # nblock = np.concatenate(np.cumsum(self.bvec.size))
        
        # istart = np.where()
        # istop = 0, 0
        raise NotImplementedError("do this later I guess")

    def slice_to_numeric(self, slice_obj, array_len):
        nstart, nstop, nstep = 0, 0, 1
        if slice_obj.start is not None:
            nstart = slice_obj.start

        if slice_obj.stop is None:
            nstop = array_len + 1
        elif slice_obj.stop < 0:
            nstop = array_len + slice_obj.stop
        else:
            nstop = slice_obj.stop

        if slice_obj.step is not None:
            nstep = slice_obj.step
        return (nstart, nstop, nstep)
