"""
This module contains the block matrix definition
"""

from typing import Tuple, List, TypeVar
import itertools
import numpy as np
from blockarray.blockarray import BlockArray
from petsc4py import PETSc

from . import subops as gops
from .labelledarray import LabelledArray, flatten_array
from .typing import (Shape, BlockShape, MultiLabels)

# pylint: disable=no-member, abstract-method
#

Icsr = List[int]
Jcsr = List[int]
Vcsr = List[float]
CSR = Tuple[Icsr, Jcsr, Vcsr]

T = TypeVar('T')
class BlockMatrix(BlockArray[T]):
    """
    Represents a block matrix with blocks indexed by keys
    """
    def __init__(self,
        array,
        shape=None,
        labels=None):
        super().__init__(array, shape, labels)

        if len(self.shape) != 2:
            raise ValueError(f"BlockMatrix must have dimension == 2, not {len(self.shape)}")

    def to_mono_petsc(self, comm=None):
        return to_mono_petsc(self, comm=comm)

    def norm(self):
        return norm(self)

    def transpose(self):
        """Return the block matrix transpose"""
        ret_labels = self.labels[::-1]
        ret_shape = self.shape[::-1]

        # Loop over the row axis last in `product` so that the row indices
        # change the fastet; this ensures that the flat tensor represent the
        # transpose
        ret_subtensors = [
            self[multi_idx[::-1]].copy().transpose()
            for multi_idx in itertools.product(
                *[range(ax_size) for ax_size in self.shape[::-1]]
                )
            ]

        return BlockMatrix(ret_subtensors, ret_shape, ret_labels)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return NotImplemented

# Utilies for constructing monolithic PETSc matrix
def to_mono_petsc(bmat: BlockArray[PETSc.Mat], comm=None, finalize: bool=True) -> PETSc.Mat:
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

    shape = bmat.shape
    bshape = bmat.bshape
    bshape_row, bshape_col = bshape

    blocks_csr = get_blocks_csr(bmat)
    i_mono, j_mono, v_mono = get_block_matrix_csr(blocks_csr, shape, bshape)

    ## Create a monolithic matrix to contain the block matrix
    ret_mat = PETSc.Mat().createAIJ(
        (np.sum(bshape_row), np.sum(bshape_col)), comm=comm
    )
    nnz = i_mono[1:] - i_mono[:-1]
    ret_mat.setUp() # You have to do this if you don't preallocate I think
    ret_mat.setPreallocationNNZ(nnz)

    ## Insert the values into the matrix
    ret_mat.setValuesCSR(i_mono, j_mono, v_mono)

    if finalize:
        ret_mat.assemble()

    return ret_mat

def get_blocks_csr(bmat: BlockArray[PETSc.Mat]) -> Tuple[List[List[Icsr]], List[List[Jcsr]], List[List[Vcsr]]]:
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
    M_BLOCK, N_BLOCK = bmat.shape

    # Grab all the CSR format values and put them into a block list form
    i_block = []
    j_block = []
    v_block = []

    for row in range(M_BLOCK):
        i_block_row = []
        j_block_row = []
        v_block_row = []
        for col in range(N_BLOCK):
            i, j, v = bmat[row, col].getValuesCSR()
            i_block_row.append(i)
            j_block_row.append(j)
            v_block_row.append(v)

        i_block.append(i_block_row)
        j_block.append(j_block_row)
        v_block.append(v_block_row)

    return i_block, j_block, v_block

def get_block_matrix_csr(blocks_csr: CSR, blocks_shape: Shape, blocks_sizes: BlockShape) -> CSR:
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

def diag_mat(n, diag=1.0, comm=None):
    diag_vec = PETSc.Vec().create(comm=comm)
    diag_vec.setSizes(n)
    diag_vec.setUp()
    diag_vec.array[:] = diag
    diag_vec.assemble()

    mat = PETSc.Mat().create(comm=comm)
    mat.setSizes([n, n])
    mat.setUp()
    mat.setDiagonal(diag_vec)
    mat.assemble()
    return mat

def ident_mat(n, comm=None):
    return diag_mat(n, diag=1.0, comm=comm)

## Basic BlockMatrix operations
def norm(A: BlockMatrix[T]):
    """
    Return the Frobenius norm of A

    Parameters
    ----------
    A : BlockMatrix
    """
    frobenius_norm = np.sum([
        gops.norm_mat(A[mm, nn])**2
        for nn in range(A.shape[1])
        for mm in range(A.shape[0])])**0.5
    return frobenius_norm

## More utilities
def concatenate_mat(bmats: List[List[BlockMatrix[T]]], labels: MultiLabels=None):
    """
    Form a block matrix by joining other block matrices

    Parameters
    ----------
    bmats : tuple(tupe(BlockMatrix))
    """
    # check the array is 2D by checking that the number of columns in each row
    # are equal, pairwise
    _, (NUM_BROW, NUM_BCOL) = flatten_array(bmats)

    mats = []
    for brow in range(NUM_BROW):
        for row in range(bmats[brow][0].shape[0]):
            mats_row = []
            for bcol in range(NUM_BCOL):
                mats_row.extend(bmats[brow][bcol][row, :])
            mats.append(mats_row)

    if labels is None:
        row_labels = [key for ii in range(NUM_BROW) for key in bmats[ii][0].labels[0]]
        col_labels = [key for jj in range(NUM_BCOL) for key in bmats[0][jj].labels[1]]
        labels = (tuple(row_labels), tuple(col_labels))
    return BlockMatrix(mats, labels=labels)

## Converting subtypes
def convert_subtype_to_petsc(bmat: BlockMatrix[T]):
    """
    Converts a block matrix from one submatrix type to the PETSc submatrix type

    Parameters
    ----------
    bmat: BlockMatrix
    """
    mats = [gops.convert_mat_to_petsc(mat) for mat in bmat.subarrays_flat]
    barray = LabelledArray(mats, bmat.shape, bmat.labels)
    return BlockMatrix(barray)
