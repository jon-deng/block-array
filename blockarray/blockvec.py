"""
This module contains the block vector definition
"""

from typing import TypeVar
import functools as ftls

import numpy as np
from petsc4py import PETSc

from . import subops as gops
from .blockarray import BlockArray
from .blockmat import BlockMatrix

## pylint: disable=no-member


# Type variable for a 'sub'vector
T = TypeVar('T')

class BlockVector(BlockArray[T]):
    """
    Represents a block vector with blocks indexed by labels
    """
    def __init__(self, barray, shape=None, labels=None):
        super().__init__(barray, shape, labels)

        if len(self.shape) > 1:
            raise ValueError(f"BlockVector must have dimension == 1, not {len(self.shape)}")

    ## Add vecs property for special case/backwards compatibilty
    @property
    def vecs(self):
        return self.subarrays_flat

    ## Basic string representation functions
    def print_summary(self):
        summary_strings = [
            f"{key}: ({np.min(vec[:])}/{np.max(vec[:])}/{np.mean(vec[:])})"
            for key, vec in self.items()]

        summary_message = ", ".join(["block: (min/max/mean)"] + summary_strings)
        print(summary_message)
        return summary_message

    def __setitem__(self, key, value):
        """
        Return the vector corresponding to the labelled block

        Parameters
        ----------
        key : str, int, slice
            A block label
        value : array_like or BlockVector
        """
        _array = self[key]
        if isinstance(_array, BlockArray):
            if isinstance(value, BlockArray):
                for subvec, subvec_value in zip(_array, value):
                    gops.set_vec(subvec, subvec_value)
            else:
                for subvec in _array:
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
        Sets all values based on a monolithic vector
        """
        # Check sizes are compatible
        assert vec.size == np.sum(self.bshape[0])

        # indices of the boundaries of each block
        n_blocks = np.concatenate(([0], np.cumsum(self.bshape[0])))
        for i, (n_start, n_stop) in enumerate(zip(n_blocks[:-1], n_blocks[1:])):
            self[i][:] = vec[n_start:n_stop]

    ## Conversion and treatment as a monolithic vector
    def to_mono_ndarray(self):
        ndarray_vecs = [np.array(vec) for vec in self]
        return np.concatenate(ndarray_vecs, axis=0)

    def to_mono_petsc_seq(self, comm=None):
        total_size = np.sum(self.mshape)
        vec = PETSc.Vec().createSeq(total_size, comm=comm)
        vec.setUp()
        vec.setArray(self.to_mono_ndarray())
        vec.assemble()
        return vec

    def to_mono_petsc(self, comm=None):
        total_size = np.sum(self.mshape)
        vec = PETSc.Vec().create(comm=comm)
        vec.setSizes(total_size)
        vec.setUp()
        vec.setArray(self.to_mono_ndarray())
        vec.assemble()
        return vec

    ## common operator overloading
    def __eq__(self, other):
        eq = False
        if isinstance(other, BlockVector):
            err = self - other
            if dot(err, err) == 0:
                eq = True
        else:
            raise TypeError(f"Cannot compare {type(other)} to {type(self)}")

        return eq

    ##
    def norm(self):
        return dot(self, self)**0.5

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return NotImplemented

def validate_blockvec_size(*args):
    """
    Check if a collection of BlockVecs have compatible block sizes
    """
    ref_bsize = args[0].bshape[0]
    valid_bsizes = [arg.bshape[0] == ref_bsize for arg in args]

    return all(valid_bsizes)

# Utilities
def split_bvec(bvec, block_sizes):
    """
    Splits a block vector into multiple block vectors
    """
    block_cumul_sizes = [0] + np.cumsum(block_sizes).tolist()
    split_bvecs = [
        bvec[ii:jj]
        for ii, jj in zip(block_cumul_sizes[:-1], block_cumul_sizes[1:])
        ]
    return tuple(split_bvecs)

def concatenate_vec(args, labels=None):
    """
    Concatenate a series of BlockVecs into a single BlockVector

    Parameters
    ----------
    args : List of BlockVector
    """
    if labels is None:
        labels = [ftls.reduce(lambda a, b: a+b, [bvec.labels[0] for bvec in args])]

    vecs = ftls.reduce(lambda a, b: a+b, [bvec.subarrays_flat for bvec in args])

    return BlockVector(vecs, labels=labels)

# Converting subtypes
def convert_subtype_to_petsc(bvec):
    """
    Converts a block matrix from one submatrix type to the PETSc submatrix type

    Parameters
    ----------
    bmat: BlockMatrix
    """
    vecs = [gops.convert_vec_to_petsc(subvec) for subvec in bvec.subarrays_flat]
    return BlockVector(vecs, labels=bvec.labels)

# Converting to monolithic vectors
def to_mono_petsc(bvec, comm=None, finalize=True):
    raise NotImplementedError()

# Converting to block matrix formats
def to_block_rowmat(bvec):
    mats = tuple([
        tuple([gops.convert_vec_to_rowmat(vec) for vec in bvec.subarrays_flat])
        ])
    return BlockMatrix(mats)

def to_block_colmat(bvec):
    mats = tuple([
        tuple([gops.convert_vec_to_colmat(vec)]) for vec in bvec.subarrays_flat
        ])
    return BlockMatrix(mats)

# Basic operations
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

def norm(a):
    """Return the 2-norm of a vector"""
    return dot(a, a)**0.5


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
