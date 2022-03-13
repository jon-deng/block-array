"""
This module contains the block matirx definition and various operations on 
block matrices
"""

import operator
from collections import OrderedDict

import numpy as np
from blocklinalg.tensor import BlockTensor
from petsc4py import PETSc

from . import genericops as gops
from .blockarray import BlockArray
from .tensor import BlockTensor

# pylint: disable=no-member

# Utilies for constructing block matrices
def form_block_matrix(blocks, finalize=True, comm=None):
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
    if comm is None:
        comm = PETSc.COMM_SELF

    blocks_shape = get_blocks_shape(blocks)
    blocks_sizes = get_blocks_sizes(blocks, blocks_shape)
    block_row_sizes, block_col_sizes = blocks_sizes

    blocks_csr = get_blocks_csr(blocks, blocks_shape)
    i_mono, j_mono, v_mono = get_block_matrix_csr(blocks_csr, blocks_shape, blocks_sizes)

    ## Create a monolithic matrix to contain the block matrix
    block_mat = PETSc.Mat()
    block_mat.create(comm)
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

# Utilities for making specific types of matrices
def zero_mat(n, m, comm=None):
    mat = PETSc.Mat().create(comm=comm)
    mat.setSizes([n, m])
    mat.setUp()
    mat.assemble()
    return mat

def ident_mat(n, comm=None):
    diag = PETSc.Vec().create(comm=comm)
    diag.setSizes(n)
    diag.setUp()
    diag.array[:] = 1
    diag.assemble()

    mat = PETSc.Mat().create(comm=comm)
    mat.setSizes([n, n])
    mat.setUp()
    mat.setDiagonal(diag)
    mat.assemble()
    return mat

## Basic BlockMat operations
def add(A, B):
    """
    Add two block matrices, A + B

    Parameters
    ----------
    A, B: BlockMat
    """
    labels = tuple([A.labels[0], B.labels[1]])
    mats = [
        [A[mm, nn] + B[mm, nn] 
        for nn in range(A.shape[0])] 
        for mm in range(A.shape[1])]
    return BlockMat(mats, labels)

def sub(A, B):
    """
    Subtract two block matrices, A - B

    Parameters
    ----------
    A, B: BlockMat
    """
    return add(A, -B)

def scalar_mul(a, B):
    """
    Scalar multiplication of B by a

    Parameters
    ----------
    a: float 
    B: BlockMat
    """
    mats = [
        [a*B[mm, nn] 
        for nn in range(B.shape[1])] 
        for mm in range(B.shape[0])]
    labels = B.labels
    return BlockMat(mats, labels)

def norm(A):
    """
    Return the Frobenius norm of A

    Parameters
    ----------
    A : BlockMat
    """
    frobenius_norm = np.sum([
        gops.norm_mat(A[mm, nn])**2
        for nn in range(A.shape[1])
        for mm in range(A.shape[0])])**0.5
    return frobenius_norm

## More utilities
def concatenate_mat(bmats):
    """
    Form a block matrix by joining other block matrices

    Parameters
    ----------
    bmats : tuple(tupe(BlockMat))
    """
    # check the array is 2D by checking that the number of columns in each row
    # are equal, pairwise
    NUM_BROW, NUM_BCOL = get_blocks_shape(bmats)

    mats = []
    for brow in range(NUM_BROW):
        for row in range(bmats[brow][0].shape[0]):
            mats_row = []
            for bcol in range(NUM_BCOL):
                mats_row.extend(bmats[brow][bcol][row, :])
            mats.append(mats_row)

    row_labels = [key for ii in range(NUM_BROW) for key in bmats[ii][0].labels[0]]
    col_labels = [key for jj in range(NUM_BCOL) for key in bmats[0][jj].labels[1]]
    labels = (tuple(row_labels), tuple(col_labels))
    return BlockMat(mats, labels)

def convert_bmat_to_petsc(bmat):
    """
    Converts a block matrix from one submatrix type to the PETSc submatrix type

    Parameters
    ----------
    bmat: BlockMat
    """
    mats = [gops.convert_mat_to_petsc(mat) for mat in bmat.array]
    barray = BlockArray(mats, bmat.shape, bmat.labels)
    return BlockMat(barray)

class BlockMat(BlockTensor):
    """
    Represents a block matrix with blocks indexed by keys

    Parameters
    ----------
    mats : tuple(tuple(PETsc.Mat))
    row keys, col_keys : tuple(str)
    """

    # @property
    # def mats(self):
    #     return self.barray.nested_array

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return scalar_mul(other, self)

    def __neg__(self):
        return scalar_mul(-1, self)

    def __pos__(self):
        return scalar_mul(1, self)

    def __radd__(self, other):
        return add(other, self)

    def __rsub__(self, other):
        return sub(other, self)

    def __rmul__(self, other):
        return scalar_mul(other, self)

    def to_petsc(self, comm=None):
        return form_block_matrix(self.array_nested)

    def norm(self):
        return norm(self)
