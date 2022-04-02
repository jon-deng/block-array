"""
Utilities for reading/writing BlockTensor objects to hdf5
"""

import h5py

from blocktensor import vec as bvec

def create_resizable_block_vector_group(
    f: h5py.Group, blocklabels, blockshape, dataset_kwargs=None):
    """
    Create a resizable datasets in a group to store the vector
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    for subvec_label, subvec_size in zip(blocklabels[0], blockshape[0]):
        f.create_dataset(
            subvec_label, (0, subvec_size), maxshape=(None, subvec_size),
            **dataset_kwargs)

def append_block_vector_to_group(f: h5py.Group, vec: bvec.BlockVec):
    """
    Append block vector data to a resizable dataset
    """
    # Loop through each block of the block vector to see if the dataset represents it
    _valid_blocks = [
        (subvec_label in f) and (subvec_size == f[subvec_label].shape[-1])
        for subvec_label, subvec_size in zip(vec.labels[0], vec.bshape[0])]
    assert all(_valid_blocks)

    for subvec_label, subvec_size, subvec in zip(vec.labels[0], vec.bshape[0], vec):
        axis0_size = f[subvec_label].shape[0] + 1
        f[subvec_label].resize(axis0_size, axis=0)
        f[subvec_label][-1, :] = subvec

def read_block_vector_from_group(f: h5py.Group, blocklabels, nvec=0):
    """
    Reads block vector data from a resizable dataset

    Parameters
    ----------
    f: h5py.Group
    blocklabels:
        A list of the labels in the block vector (from the .labels attribute of
        BlockVec)
    nvec: int
        The index of the block vector
    """
    subvecs = [f[block_label][nvec, :] for block_label in blocklabels[0]]
    return bvec.BlockVec(subvecs, (len(subvecs),), blocklabels)