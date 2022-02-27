"""
This module contains the block vector definition and various operations on 
block vectors
"""

import operator
from collections import OrderedDict
from typing import TypeVar, Generic, List, Optional

import numpy as np
from petsc4py import PETSc

from .genericops import set_vec

# Type variable for a 'sub'vector
T = TypeVar('T')

# BlockVec methods
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

        bsize = bvecs[0].bsize
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

def generic_vec_size(vec):
    if isinstance(vec, np.ndarray):
        return vec.size
    elif isinstance(vec, float):
        return 1
    else:
        return len(vec)

class BlockVec(Generic[T]):
    """
    Represents a block vector with blocks indexed by keys

    Parameters
    ----------
    vecs : tuple(PETsc.Vec or dolfin.cpp.la.PETScVec or np.ndarray)
    keys : tuple(str)
    """
    def __init__(self, vecs: List[T], keys: Optional[List[str]]=None):
        if keys is None:
            keys = tuple(range(len(vecs)))

        self._keys = tuple(keys)
        self._vecs = tuple(vecs)
        self.data = dict(zip(keys, vecs))

    @property
    def size(self):
        """Return sizes of each block"""
        return tuple([generic_vec_size(vec) for vec in self.vecs])

    @property
    def bsize(self):
        """
        Block size of the vector (number of blocks)
        """
        return len(self.vecs)

    @property
    def keys(self):
        return self._keys

    @property
    def vecs(self):
        """Return tuple of vectors from each block"""
        return self._vecs

    def copy(self):
        """Return a copy of the block vector"""
        keys = self.keys
        vecs = tuple([vec.copy() for vec in self.vecs])
        return type(self)(vecs, keys)

    def __copy__(self):
        return self.copy()
    
    ## Basic string representation functions
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        desc = ", ".join([f"{key}:{generic_vec_size(vec)}" for key, vec in zip(self.keys, self.vecs)])
        return f"({desc})"

    ## Dict-like interface
    def __contains__(self, key):
        return key in self.data

    def __iter__(self):
        for key in self.keys:
            yield self.data[key]

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

    def __getitem__(self, key):
        """
        Return the vector or BlockVec corresponding to the index

        Parameters
        ----------
        key : str, int, slice
            A block label
        """
        if isinstance(key, str):
            try:
                return self.data[key]
            except KeyError:
                raise KeyError(f"`{key}` is not a valid block key")
        elif isinstance(key, int):
            try:
                return self.vecs[key]
            except IndexError as e:
                raise e
        elif isinstance(key, slice):
            vecs = self.vecs[key]
            keys = self.keys[key]
            return BlockVec(vecs, keys)
        else:
            raise TypeError(f"`{key}` must be either str, int or slice")
    
    def __setitem__(self, key, value):
        """
        Return the vector corresponding to the labelled block

        Parameters
        ----------
        key : str, int, slice
            A block label
        value : array_like or BlockVec
        """
        if isinstance(key, str):
            if key in self.data:
                self.data[key][:] = value
            else:
                raise KeyError(f"`{key}` is not a valid block key")
        elif isinstance(key, int):
            try:
                self.vecs[key][:] = value
            except IndexError as e:
                raise e
        elif isinstance(key, slice):
            assert isinstance(value, BlockVec)
            assert self.size[key] == value.size
            
            for vec, val, name in zip(self.vecs[key], value.vecs, self.keys[key]):
                vec[:] = val
        else:
            raise TypeError(f"`{key}` must be either str, int or slice")

    def set(self, scalar):
        """
        Set a constant value for the block vector
        """
        for vec in self:
            set_vec(vec, scalar)

    def set_vec(self, vec):
        """
        Sets all values based on a vector
        """
        # Check sizes are compatible
        assert vec.size == np.sum(self.size)

        # indices of the boundaries of each block
        n_blocks = np.concatenate(([0], np.cumsum(self.size)))
        for i, (n_start, n_stop) in enumerate(zip(n_blocks[:-1], n_blocks[1:])):
            self[i][:] = vec[n_start:n_stop]

    ## Conversion and treatment as a monolithic vector
    def to_ndarray(self):
        ndarray_vecs = [np.array(vec) for vec in self.vecs]
        return np.concatenate(ndarray_vecs, axis=0)

    def to_petsc_seq(self, comm=None):
        total_size = np.sum(self.size)
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
