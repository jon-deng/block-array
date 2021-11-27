"""
This module contains various utilities for sparse linear algebra
"""

import operator
from collections import OrderedDict

import numpy as np
from petsc4py import PETSc

def form_block_matrix(blocks, finalize=True):
    """
    Form a monolithic block matrix by combining matrices in `blocks`

    Parameters
    ----------
    blocks : [[Petsc.Mat, ...]]
        A list of lists containing the matrices forming the blocks of the desired block matrix.
        These are organized as:
            [[mat00, ..., mat0n],
             [mat10, ..., mat1n],
             [  ..., ...,   ...],
             [matm0, ..., matmn]]
    """
    blocks_shape = get_blocks_shape(blocks)
    blocks_sizes = get_blocks_sizes(blocks, blocks_shape)
    block_row_sizes, block_col_sizes = blocks_sizes

    blocks_csr = get_blocks_csr(blocks, blocks_shape)
    i_mono, j_mono, v_mono = get_block_matrix_csr(blocks_csr, blocks_shape, blocks_sizes)

    ## Create a monolithic matrix to contain the block matrix
    block_mat = PETSc.Mat()
    block_mat.create(PETSc.COMM_SELF)
    block_mat.setSizes([np.sum(block_row_sizes), np.sum(block_col_sizes)])

    block_mat.setUp() # You have to do this if you don't preallocate I think

    ## Insert the values into the matrix
    nnz = i_mono[1:] - i_mono[:-1]
    block_mat.setPreallocationNNZ(nnz)
    block_mat.setValuesCSR(i_mono, j_mono, v_mono)

    if finalize:
        block_mat.assemble()

    return block_mat

def get_blocks_shape(blocks):
    """
    Return the shape of the block matrix, and the sizes of each block

    The function will also check if the supplied blocks have consistent shapes for a valid block
    matrix.

    Parameters
    ----------
    blocks : [[Petsc.Mat, ...]]
        A list of lists containing the matrices forming the blocks of the desired block matrix.
        These are organized as:
            [[mat00, ..., mat0n],
             [mat10, ..., mat1n],
             [  ..., ...,   ...],
             [matm0, ..., matmn]]
    """
    ## Get the block sizes
    # also check that the same number of columns are supplied in each row
    M_BLOCK = len(blocks)
    N_BLOCK = len(blocks[0])
    for row in range(1, M_BLOCK):
        assert N_BLOCK == len(blocks[row])

    return M_BLOCK, N_BLOCK

def get_blocks_sizes(blocks, blocks_shape):
    """
    Return the sizes of each block in the block matrix

    Parameters
    ----------
    blocks : [[Petsc.Mat, ...]]
        A list of lists containing the matrices forming the blocks of the desired block matrix.
        These are organized as:
            [[mat00, ..., mat0n],
             [mat10, ..., mat1n],
             [  ..., ...,   ...],
             [matm0, ..., matmn]]
    blocks_shape : (M, N)
        The shape of the blocks matrix i.e. number of row blocks by number of column blocks.

    Returns
    -------
    block_row_sizes, block_col_sizes: np.ndarray
        An array containing the number of rows/columns in each row/column block. For example, if
        the block matrix contains blocks of shape
        [[(2, 5), (2, 6)],
         [(7, 5), (7, 6)]]
        `block_row_sizes` and `block_col_sizes` will be [2, 7] and [5, 6], respectively.
    """
    M_BLOCK, N_BLOCK = blocks_shape

    ## Calculate an array containing the n_rows in each row block
    # and n_columns in each column block
    block_row_sizes = -1*np.ones(M_BLOCK, dtype=np.intp)
    block_col_sizes = -1*np.ones(N_BLOCK, dtype=np.intp)

    # check that row/col sizes are consistent with the other row/col block sizes
    for row in range(M_BLOCK):
        for col in range(N_BLOCK):
            block = blocks[row][col]
            shape = None
            if isinstance(block, PETSc.Mat):
                shape = block.getSize()
            elif isinstance(block, (int, float)):
                # Use -1 to indicate a variable size 'diagonal' matrix, that will adopt
                # the shape of neighbouring blocks to form a proper block matrix
                shape = (-1, -1)
            else:
                raise ValueError("Blocks can only be matrices or floats")

            for block_sizes in (block_row_sizes, block_col_sizes):
                if block_sizes[row] == -1:
                    block_sizes[row] = shape[0]
                else:
                    assert (block_sizes[row] == shape[0]
                            or shape[0] == -1)

    # convert any purely variable size blocks to size 1 blocks
    block_row_sizes = np.where(block_row_sizes == -1, 1, block_row_sizes)
    block_col_sizes = np.where(block_col_sizes == -1, 1, block_col_sizes)
    return block_row_sizes, block_col_sizes

def get_blocks_csr(blocks, blocks_shape):
    """
    Return the CSR format data for each block in a block matrix form

    Parameters
    ----------
    blocks : [[Petsc.Mat, ...]]
        A list of lists containing the matrices forming the blocks of the desired block matrix.
        These are organized as:
            [[mat00, ..., mat0n],
             [mat10, ..., mat1n],
             [  ..., ...,   ...],
             [matm0, ..., matmn]]
    blocks_shape : (M, N)
        The shape of the blocks matrix i.e. number of row blocks by number of column blocks.

    Returns
    -------
    [[(i, j, v), ...]]
        A 2d list containing the CSR data for each block
    """
    M_BLOCK, N_BLOCK = blocks_shape

    # Grab all the CSR format values and put them into a block list form
    i_block = []
    j_block = []
    v_block = []

    for row in range(M_BLOCK):
        i_block_row = []
        j_block_row = []
        v_block_row = []
        for col in range(N_BLOCK):
            block = blocks[row][col]
            if isinstance(block, PETSc.Mat):
                i, j, v = block.getValuesCSR()
                i_block_row.append(i)
                j_block_row.append(j)
                v_block_row.append(v)
            else:
                # In this case the block should just be a constant value, like 1.0
                # to indicate an identity matrix
                i_block_row.append(None)
                j_block_row.append(None)
                v_block_row.append(block)
        i_block.append(i_block_row)
        j_block.append(j_block_row)
        v_block.append(v_block_row)

    return i_block, j_block, v_block

def get_block_matrix_csr(blocks_csr, blocks_shape, blocks_sizes):
    """
    Return csr data associated with monolithic block matrix

    Parameters
    ----------
    blocks : [[Petsc.Mat, ...]]
        A list of lists containing the matrices forming the blocks of the desired block matrix.
        These are organized as:
            [[mat00, ..., mat0n],
             [mat10, ..., mat1n],
             [  ..., ...,   ...],
             [matm0, ..., matmn]]
    blocks_shape : (M, N)
        The shape of the blocks matrix i.e. number of row blocks by number of column blocks.
    blocks_sizes : [], []
        A tuple of array containing the number of rows in each row block, and the number of columns
        in each column block.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        A tuple of I, J, V CSR data for the monolithic matrix corresponding to the supplied blocks
    """
    i_block, j_block, v_block = blocks_csr
    M_BLOCK, N_BLOCK = blocks_shape
    block_row_sizes, block_col_sizes = blocks_sizes

    # block_row_offsets = np.concatenate(([0] + np.cumsum(block_row_sizes)[:-1]))
    block_col_offsets = np.concatenate(([0], np.atleast_1d(np.cumsum(block_col_sizes)[:-1])))

    i_mono = [0]
    j_mono = []
    v_mono = []

    for row in range(M_BLOCK):
        for local_row in range(block_row_sizes[row]):
            j_mono_row = []
            v_mono_row = []
            for col in range(N_BLOCK):
                # get the CSR data associated with the specific block
                i, j, v = i_block[row][col], j_block[row][col], v_block[row][col]

                # if the block is not a matrix, handle this case as if it's a diagonal block
                if i is None:
                    if v != 0 or row == col:
                        # only set the 'diagonal' if the matrix dimension is appropriate
                        # i.e. if the block is a tall rectangular one, don't keep
                        # writing diagonals when the row index > # cols since this is
                        # undefined
                        if local_row < block_col_sizes[col]:
                            j_mono_row += [local_row + block_col_offsets[col]]
                            v_mono_row += [v]
                else:
                    istart = i[local_row]
                    iend = i[local_row+1]

                    j_mono_row += (j[istart:iend] + block_col_offsets[col]).tolist()
                    v_mono_row += v[istart:iend].tolist()
            i_mono += [i_mono[-1] + len(v_mono_row)]
            j_mono += j_mono_row
            v_mono += v_mono_row

    i_mono = np.array(i_mono, dtype=np.int32)
    j_mono = np.array(j_mono, dtype=np.int32)
    v_mono = np.array(v_mono, dtype=np.float)
    return i_mono, j_mono, v_mono

def reorder_mat_rows(mat, rows_in, rows_out, m_out, finalize=True):
    """
    Reorder rows of a matrix to a new matrix with a possibly different number of rows

    This is useful for transforming matrices where data is shared between domains along an
    interface. The rows corresponding to the interface on domain 1 have to be mapped to the rows on
    the interface in domain 2.

    The number of columns is presevered between the two matrices.
    """
    # sort the output indices in increasing row index
    is_sort = np.argsort(rows_out)
    rows_in = rows_in[is_sort]
    rows_out = rows_out[is_sort]

    m_in, n_in = mat.getSize()
    i_in, j_in, v_in = mat.getValuesCSR()

    i_out = [0]
    j_out = []
    v_out = []
    row_out_prev = 0
    for row_in, row_out in zip(rows_in, rows_out):
        idx_start = i_in[row_in]
        idx_end = i_in[row_in+1]

        # Add zero rows to the array
        i_out += [i_out[-1]]*max((row_out-row_out_prev-1), 0) # max function ensures no zero rows added if row_out==0

        # Add the nonzero row components
        i_out.append(i_out[-1]+idx_end-idx_start)
        j_out += j_in[idx_start:idx_end].tolist()
        v_out += v_in[idx_start:idx_end].tolist()

        row_out_prev = row_out

    i_out = np.array(i_out, dtype=np.int32)
    j_out = np.array(j_out, dtype=np.int32)
    v_out = np.array(v_out, dtype=np.float64)

    mat_out = PETSc.Mat()
    mat_out.create(PETSc.COMM_SELF)
    mat_out.setSizes([m_out, n_in])

    nnz = i_out[1:]-i_out[:-1]
    mat_out.setUp()
    mat_out.setPreallocationNNZ(nnz)

    mat_out.setValuesCSR(i_out, j_out, v_out)

    if finalize:
        mat_out.assemble()

    return mat_out

def reorder_mat_cols(mat, cols_in, cols_out, n_out, finalize=True):
    """
    Reorder columns of a matrix to a new one with a possibly different number of columns

    Parameters
    ----------
    mat : Petsc.Mat
    cols_in : array[]
        columns indices of input matrix
    cols_out : array
        column indices of output matrix
    n_out :
        number of columns in output matrix
    """
    # Sort the column index permutations
    # is_sort = np.argsort(cols_out)
    # cols_in = cols_in[is_sort]
    # cols_out = cols_out[is_sort]
    col_in_to_out = dict([(j_in, j_out) for j_in, j_out in zip(cols_in, cols_out)])

    # insert them into a dummy size array that's used to

    m_in, n_in = mat.getSize()
    i_in, j_in, v_in = mat.getValuesCSR()
    
    # assert n_out >= n_in

    i_out = [0]
    j_out = []
    v_out = []
    for row in range(m_in):
        idx_start = i_in[row]
        idx_end = i_in[row+1]

        j_in_row = j_in[idx_start:idx_end]
        v_in_row = v_in[idx_start:idx_end]

        # build the column and value csr components for the row
        j_out_row = [col_in_to_out[j] for j in j_in_row if j in col_in_to_out]
        v_out_row = [v for j, v in zip(j_in_row, v_in_row.tolist()) if j in col_in_to_out]

        # add them to the global j, v csr arrays
        j_out += j_out_row
        v_out += v_out_row
        i_out.append(i_out[-1] + len(j_out_row))

    i_out = np.array(i_out, dtype=np.int32)
    j_out = np.array(j_out, dtype=np.int32)
    v_out = np.array(v_out, dtype=np.float64)

    mat_out = PETSc.Mat()
    mat_out.create(PETSc.COMM_SELF)
    mat_out.setSizes([m_in, n_out])

    nnz = i_out[1:]-i_out[:-1]
    mat_out.setUp()
    mat_out.setPreallocationNNZ(nnz)

    mat_out.setValuesCSR(i_out, j_out, v_out)

    if finalize:
        mat_out.assemble()

    return mat_out

# BlockVec methods
def general_vec_set(vec, vals):
    """
    Set the specified values to a vector

    This handles the different indexing notations that vectors can have based on the type, shape,
    etc.

    Parameters
    ----------
    vec : PETScVec, PETsc.Vec, np.ndarray
        One of the valid types that can be contained in a BlockVec
    vals : float, array_like, etc.
    """
    if isinstance(vec, np.ndarray) and vec.shape == ():
        vec[()] = vals
    else:
        vec[:] = vals

def concatenate(*args):
    """
    Concatenate a series of BlockVecs into a single BlockVec

    Parameters
    ----------
    args : BlockVec
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

def _len(vec):
    if isinstance(vec, np.ndarray):
        return vec.size
    elif isinstance(vec, float):
        return 1
    else:
        return len(vec)

class BlockVec:
    """
    Represents a block vector with blocks indexed by keys

    Parameters
    ----------
    vecs : tuple(PETsc.Vec or dolfin.cpp.la.PETScVec or np.ndarray)
    keys : tuple(str)
    """
    def __init__(self, vecs, keys=None):
        if keys is None:
            keys = tuple(range(len(vecs)))

        self._keys = tuple(keys)
        self._vecs = tuple(vecs)
        self.data = dict(zip(keys, vecs))

    @property
    def size(self):
        """Return sizes of each block"""
        return tuple([_len(vec) for vec in self.vecs])

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
        desc = ", ".join([f"{key}:{_len(vec)}" for key, vec in zip(self.keys, self.vecs)])
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
            general_vec_set(vec, scalar)

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

class BlockMat:
    """
    A block matrix
    """
