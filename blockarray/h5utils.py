"""
Utilities for reading/writing BlockArray objects to hdf5
"""

from . import _HAS_H5PY, require_h5py
if _HAS_H5PY:
    import h5py
else:
    h5py = None

from blockarray import blockvec as bvec

@require_h5py
def create_resizable_block_vector_group(
    f: 'h5py.Group', blocklabels, blockshape, dataset_kwargs=None):
    """
    Create a resizable datasets in a group to store the vector
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    # Set group attributes to store the block vector:
    # shape, labels and dimension
    tensor_shape = [len(axis_sizes) for axis_sizes in blockshape]
    f.attrs.create('blocktensor_shape', tensor_shape)
    f.attrs.create('blocktensor_dim', len(tensor_shape))
    for naxis in range(len(blockshape)):
        f.attrs.create(f'blocktensor_axis{naxis}_labels', blocklabels[naxis])

    for subvec_label, subvec_size in zip(blocklabels[0], blockshape[0]):
        f.create_dataset(
            subvec_label, (0, subvec_size), maxshape=(None, subvec_size),
            **dataset_kwargs)

@require_h5py
def append_block_vector_to_group(f: 'h5py.Group', vec: bvec.BlockVector):
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

@require_h5py
def read_block_vector_from_group(f: 'h5py.Group', nvec=0):
    """
    Reads block vector data from a resizable dataset

    Parameters
    ----------
    f: h5py.Group
    nvec: int
        The index of the block vector
    """
    shape = tuple(f.attrs['blocktensor_shape'])
    ndim = f.attrs['blocktensor_dim']
    labels = [
        tuple(f.attrs[f'blocktensor_axis{naxis}_labels'])
        for naxis in range(ndim)]

    subvecs = [f[block_label][nvec, :] for block_label in labels[0]]
    return bvec.BlockVector(subvecs, shape, labels)

