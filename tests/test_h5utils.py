"""
Tests that h5utils works
"""

import numpy as np
import h5py

from blocktensor import vec as bvec
from blocktensor.h5utils import (
    create_resizable_block_vector_group,
    append_block_vector_to_group,
    read_block_vector_from_group)

def setup_block_vecs():
    """
    Return a list of block vecs
    """
    labels = [['a', 'b', 'c']]
    subvec_sizes = [1, 2, 3]

    bvecs = []
    for offset in range(5):
        subvecs = [np.arange(size) for size in subvec_sizes]
        bvecs.append(bvec.BlockVector(subvecs, (len(subvec_sizes),), labels))

    return bvecs

def test_create_resizable_block_vector_group(bvec):
    with h5py.File("test_create.h5", mode='w') as f:
        create_resizable_block_vector_group(f, bvec.labels, bvec.bshape)

        h5_subvec_sizes = [f[label].shape[-1] for label in bvec.labels[0]]
        bvec_subvec_sizes = bvec.bshape[0]

        valid_sizes = [
            h5_subvec_size == bvec_subvec_size
            for h5_subvec_size, bvec_subvec_size
            in zip(h5_subvec_sizes, bvec_subvec_sizes)]
        assert all(valid_sizes)

def test_append_block_vector_to_group(blockvectors):
    with h5py.File("test_create.h5", mode='w') as f:
        _vec = blockvectors[0]
        create_resizable_block_vector_group(f, _vec.labels, _vec.bshape)

        for blockvector in blockvectors:
            append_block_vector_to_group(f, blockvector)

        for ii, blockvector in enumerate(blockvectors):
            subvecs_ref = [subvec for subvec in blockvector]
            subvecs_h5 = [f[label][ii, :] for label in blockvector.labels[0]]
            subvecs_valid = [
                np.all(subvec_ref == subvec_h5)
                for subvec_ref, subvec_h5 in zip(subvecs_ref, subvecs_h5)]
            assert all(subvecs_valid)

def test_read_block_vector_from_group(blockvectors):
    with h5py.File("test_create.h5", mode='w') as f:
        _vec = blockvectors[0]
        create_resizable_block_vector_group(f, _vec.labels, _vec.bshape)

        for blockvector in blockvectors:
            append_block_vector_to_group(f, blockvector)

            blockvector_h5 = read_block_vector_from_group(f, -1)
            blockvector_ref = blockvector

            assert (blockvector_h5-blockvector_ref).norm() == 0

if __name__ == '__main__':
    bvecs = setup_block_vecs()

    test_create_resizable_block_vector_group(bvecs[0])
    test_append_block_vector_to_group(bvecs)

    test_read_block_vector_from_group(bvecs)