"""
Tests that h5utils works
"""

import pytest

import numpy as np
import h5py

from blockarray import blockvec as bvec
from blockarray.h5utils import (
    create_resizable_block_vector_group,
    append_block_vector_to_group,
    read_block_vector_from_group,
)


@pytest.fixture()
def setup_bvec():
    """
    Return a `BlockVector` instance
    """
    labels = (('a', 'b', 'c'),)
    subvecs = (1 * np.ones(2), 4 * np.ones(10), 2 * np.ones(3))
    return bvec.BlockVector(subvecs, labels=labels)


def test_create_resizable_block_vector_group(setup_bvec, tmp_path):
    vec = setup_bvec
    with h5py.File(f"{tmp_path/'test.h5'}", mode='w') as f:
        create_resizable_block_vector_group(f, vec.labels, vec.bshape)

        h5_subvec_sizes = [f[label].shape[-1] for label in vec.labels[0]]
        bvec_subvec_sizes = vec.bshape[0]

        valid_sizes = [
            h5_subvec_size == bvec_subvec_size
            for h5_subvec_size, bvec_subvec_size in zip(
                h5_subvec_sizes, bvec_subvec_sizes
            )
        ]
        assert all(valid_sizes)


def test_append_block_vector_to_group(setup_bvec, tmp_path):
    vec = setup_bvec
    vecs = [i * setup_bvec.copy() for i in range(1, 4)]

    with h5py.File(f"{tmp_path/'test.h5'}", mode='w') as f:
        create_resizable_block_vector_group(f, vec.labels, vec.bshape)

        for blockvector in vecs:
            append_block_vector_to_group(f, blockvector)

        for ii, blockvector in enumerate(vecs):
            subvecs_ref = [subvec for subvec in blockvector]
            subvecs_h5 = [f[label][ii, :] for label in blockvector.labels[0]]
            subvecs_valid = [
                np.all(subvec_ref == subvec_h5)
                for subvec_ref, subvec_h5 in zip(subvecs_ref, subvecs_h5)
            ]
            assert all(subvecs_valid)


def test_read_block_vector_from_group(setup_bvec, tmp_path):
    vec = setup_bvec
    vecs = [i * setup_bvec.copy() for i in range(1, 4)]

    with h5py.File(f"{tmp_path/'test.h5'}", mode='w') as f:
        create_resizable_block_vector_group(f, vec.labels, vec.bshape)

        for blockvector in vecs:
            append_block_vector_to_group(f, blockvector)

            blockvector_h5 = read_block_vector_from_group(f, -1)
            blockvector_ref = blockvector

            assert (blockvector_h5 - blockvector_ref).norm() == 0
