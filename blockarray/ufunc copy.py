"""
Module implementing `ufunc` logic
"""

import operator
from numbers import Number
import itertools
from typing import Tuple, List, Mapping, Optional
import numpy as np

from . import blockarray as ba
from . import typing

Signature = Tuple[str, ...]
Signatures = List[Signature]

Shapes = List[typing.Shape]

def parse_ufunc_signature(
        sig: str
    ) -> Tuple[Signatures, Signatures]:
    """
    Parse a ufunc signature into a nicer format

    For a ufunc signature string
    '(i,j),(j,k)->(i,k)'
    this function represents the inputs and output axis labels in a tuple
    `('i', 'j') ('j', 'k') ('i', 'k')`
    """
    # split into input and output signatures
    sig = sig.replace(' ', '')
    sig_inputs, sig_outputs = sig.split('->')

    # further split the input/output signatures into signatures for each
    # input/output
    sig_inputs = sig_inputs.split('),(')
    sig_inputs[0] = sig_inputs[0].replace('(', '')
    sig_inputs[-1] = sig_inputs[-1].replace(')', '')

    sig_outputs = sig_outputs.split('),(')
    sig_outputs[0] = sig_outputs[0].replace('(', '')
    sig_outputs[-1] = sig_outputs[-1].replace(')', '')

    # Change the signatures into tuples of symbols
    sig_inputs = [tuple(sig_input.split(',')) for sig_input in sig_inputs]
    sig_outputs = [tuple(sig_output.split(',')) for sig_output in sig_outputs]
    return sig_inputs, sig_outputs

def interpret_ufunc_signature(
        sig_ins: Signatures,
        sig_outs: Signatures
    ) -> Tuple[
        Mapping[str, Tuple[int, int]],
        Mapping[str, List[Tuple[int, int]]]
    ]:
    """
    Interprets a ufunc signature

    Parameters
    ----------
    sig_ins, sig_outs:
        Signatures for inputs and outputs

    Returns
    -------
    free_dname_to_in: Dict
        A description of free dimension names. This
        dictionary maps the dimension name to a tuple of 2 integers `(nin, dim)`
        containing the input number and dimension of the dimension name. For
        example, a signature '(i,j),(j,k)->(i,k)' has free dimension names of
        'i,k' and would have `free_dname_descr` be
        `{'i': (0, 0), 'k': (1, 1)}`.
    redu_dname_to_ins: Dict
        A description of reduced dimension names.
        This dictionary maps the dimension name to a list of tuples of 2
        integers `(nin, dim)` containing inputs and dimensions where the reduced
        dimension occurs. For example, a signature '(i,j),(j,k)->(i,k)' has
        reduced dimension names of 'j' and would have `redu_dname_descr` be
        `{'j': [(0, 1), (1, 0)]}`.
    """
    # TODO: Will have to handle weird signatures where output dimension names
    # do not match and of the input dimension names

    # Get the set of free dimension names and contract (cont) dimension names
    free_names = {name for sig_out in sig_outs for name in sig_out}
    redu_names = {
        name for sig_in in sig_ins for name in sig_in
        if name not in free_names
    }

    # For each free dimension name, record the input number and axis number that
    # it occurs in
    free_dname_to_in = {
        name: (ii_input, ii_ax)
        for ii_input, sig_input in enumerate(sig_ins)
        for ii_ax, name in enumerate(sig_input)
        if name in free_names
    }
    assert set(free_dname_to_in.keys()) == free_names

    # For each reduced dimension name, record the axis indices it occurs in
    # for each input
    redu_dname_to_ins = {name: [] for name in list(redu_names)}
    for ii_input, sig_input in enumerate(sig_ins):
        for ii_ax, name in enumerate(sig_input):
            if name in redu_dname_to_ins:
                redu_dname_to_ins[name].append(tuple([ii_input, ii_ax]))

    return free_dname_to_in, redu_dname_to_ins

def split_shapes_by_signatures(
        shapes: List[typing.Shape],
        sigs: Signatures,
        axes: List[Tuple[int, ...]]
    ) -> Tuple[Shapes, Shapes]:
    """
    Splits a list of shapes into lists of elementwise dims and core dims

    Parameters
    ----------
    shapes : 
        a list of shapes
    sigs :
        a list of signatures, one for each shape
    axes :
        a list of axes arguments, one for each shape. Only accepts positive axis 
        arguments
    """
    # Convert the shape tuples to lists so that we can pop off core dimensions
    # and have the elementwise dimensions remaining

    core_shapes = [
        tuple([shape[ax] for ax in axs])
        for shape, axs in zip(shapes, axes)
    ]
    ewise_shapes = [
        tuple([
            ax_size for ii, ax_size in enumerate(shape) 
            if ii not in set(axs)
        ])
        for shape, axs in zip(shapes, axes)
    ]
    # ewise_shapes = [tuple(shape) for shape in _shapes]
    return ewise_shapes, core_shapes

def calculate_output_shapes(
        e_shape_ins: Shapes,
        c_shape_ins: Shapes,
        sig_ins: Signatures,
        sig_outs: Signatures,
        free_name_to_in : Optional[Mapping[str, Tuple[int, int]]]=None
    ) -> Tuple[Shapes, Shapes]:
    """
    Calculate the shape of the output BlockArray
    """
    # Check that the element wise dims of all inputs are the same
    # TODO: support broadcasting?
    for shapea, shapeb in zip(e_shape_ins[:-1], e_shape_ins[1:]):
        assert shapea == shapeb

    if free_name_to_in is None:
        free_name_to_in, _ = interpret_ufunc_signature(sig_ins, sig_outs)

    eshape_out = e_shape_ins[0]
    eshape_outs = [eshape_out] * len(sig_outs)

    cshape_outs = [
        tuple([
            c_shape_ins[free_name_to_in[label][0]][free_name_to_in[label][1]]
            for label in sig
        ])
        for sig in sig_outs
    ]

    return eshape_outs, cshape_outs

def make_gen_in_multi_index(
        e_ndim_ins: List[int],
        sig_ins: Signatures,
        sig_out: Signature
    ):
    """
    Make a function that generates indices for inputs given an output index
    """
    free_name_to_output = {label: ii for ii, label in enumerate(sig_out)}

    def gen_in_multi_index(out_multi_idx):
        e_midx_outs = out_multi_idx[:-len(sig_out)]
        c_midx_outs = out_multi_idx[-len(sig_out):]

        e_midx_ins = [e_midx_outs[-n:] for n in e_ndim_ins]
        c_midx_ins = [
            tuple([
                c_midx_outs[free_name_to_output[label]]
                if label in free_name_to_output
                else slice(None)
                for label in sig_input
            ])
            for sig_input in sig_ins
        ]

        midx_ins = [
            ewise+core for ewise, core in zip(e_midx_ins, c_midx_ins)
        ]
        return midx_ins

    return gen_in_multi_index

def recursive_concatenate(arrays: typing.FlatArray, shape: typing.Shape, axes: typing.StdIndices):
    """
    Recursively concatenate logically nested list of arrays
    """
    assert len(arrays) == np.prod(shape)
    N = len(arrays)

    ret_arrays = arrays
    for ax_size, axis in zip(shape[::-1], axes[::-1]):
        concat_arrays = [
            ret_arrays[n*ax_size:(n+1)*ax_size]
            for n in range(len(ret_arrays)//ax_size)
        ]
        ret_arrays = [np.concatenate(arrays, axis) for arrays in concat_arrays]

    assert len(ret_arrays) == 1
    return ret_arrays[0]

def apply_permutation(perm: List[int], arg):
    """
    Apply a permutation to supplied lists
    """
    # check the permutation is valid
    assert max(perm) == len(perm) - 1

    # check the list to permute is valid
    assert len(arg) == len(perm)

    return type(arg)([arg[ii] for ii in perm])

def conv_neg(n, size):
    if n < 0:
        return size-n
    else:
        return n

def apply_ufunc_array(ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    Apply a ufunc on sequence of BlockArray inputs
    """

    if method != '__call__':
        return NotImplemented

    # Check input types
    if not all([
            isinstance(input, (Number, ba.BlockArray)) 
            for input in inputs
        ]):
        input_types = [type(x) for x in inputs]
        raise TypeError(f"Inputs must be of type `scalar` or `BlockArray`, not {input_types}")
    
    # Parse signature into nice format
    if ufunc.signature is None:
        signature = ','.join(['()']*ufunc.nin) + '->' + ','.join(['()']*ufunc.nout)
    else:
        signature = ufunc.signature
    sig_ins, sig_outs = parse_ufunc_signature(signature)

    # Remove any scalar inputs from the list of inputs/signatures
    # These should be put back in when the ufunc is computed on subarrays
    scalar_descr_inputs = [(ii, x) for ii, x in enumerate(inputs) if isinstance(x, Number)]
    inputs = [x for x in inputs if not isinstance(x, Number)]
    sig_ins = [sig for x, sig in zip(inputs, sig_ins) if not isinstance(x, Number)]

    free_name_to_in, redu_name_to_in = interpret_ufunc_signature(sig_ins, sig_outs)

    ## Compute a permutation of the shape from the axes kwargs
    ndim_ins = [input.ndim for input in inputs]
    _ndim_ewise_ins = [ndim-len(sig) for ndim, sig in zip(ndim_ins, sig_ins)]
    ndim_outs = [max(_ndim_ewise_ins)+len(sig) for sig in sig_outs]
    ndims = ndim_ins + ndim_outs

    if 'axes' in kwargs:
        axes = kwargs['axes']
        axes = [
            tuple([conv_neg(ii, ndim) for ii in axs]) 
            for ndim, axs in zip(ndim_ins+ndim_outs, axes)
        ]
    else:
        axes = [
            tuple([
                ndim-ii for ii in range(len(sig), 0, -1)
            ]) 
            for ndim, sig in zip(ndim_ins+ndim_outs, sig_ins+sig_outs)
        ]
    axes_ins = axes[:ufunc.nin]
    axes_outs = axes[ufunc.nin:]

    # Compute the shape permutation from axes
    # This permutes the axis sizes in shape so the core dimensions are at the end
    # and elementwise dimensions are at the beginning
    permuts = [
        tuple([ii for ii in range(ndim) if ii not in set(axes)]) + axs
        for axs, ndim in zip(axes, ndims)
    ]
    permut_ins = permuts[:ufunc.nin]
    permut_outs = permuts[ufunc.nin:]

    shape_ins = [
        apply_permutation(input.shape, perm) 
        for input, perm in zip(inputs, permut_ins)
    ]
    eshape_ins, cshape_ins = split_shapes_by_signatures(shape_ins, sig_ins, axes_ins)
    e_ndim_ins = [len(eshape) for eshape in eshape_ins]

    eshape_outs, cshape_outs = calculate_output_shapes(
        eshape_ins, cshape_ins, sig_ins, sig_outs, free_name_to_in
    )

    elabels_ins = [input.labels[:-len(sig_in)] for input, sig_in in zip(inputs, sig_ins)]
    clabels_ins = [input.labels[-len(sig_in):] for input, sig_in in zip(inputs, sig_ins)]
    clabels_outs = [
        tuple([
            clabels_ins[free_name_to_in[name][0]][free_name_to_in[name][1]]
            for name in sig_out
        ])
        for sig_out in sig_outs
    ]
    labels_outs = [elabels_ins[0] + clabels_out for clabels_out in clabels_outs]

    shape_outs = [eshape+cshape for eshape, cshape in zip(eshape_outs, cshape_outs)]
    outputs = []
    for shape_out, labels_out, sig_out in zip(shape_outs, labels_outs, sig_outs):
        gen_in_midx = make_gen_in_multi_index(e_ndim_ins, sig_ins, sig_out)

        subarrays_out = []
        for midx_out in itertools.product(
            *[range(ax_size) for ax_size in shape_out]
        ):
            midx_ins = gen_in_midx(midx_out)
            subarray_ins = [
                input[midx_in] for input, midx_in in zip(inputs, midx_ins)
            ]
            subarray_ins = [
                subarray.to_mono_ndarray()
                if isinstance(subarray, ba.BlockArray)
                else subarray
                for subarray in subarray_ins
            ]
            # Put any scalar inputs back into subarray_ins
            for ii, scalar in scalar_descr_inputs:
                subarray_ins.insert(ii, scalar)

            subarrays_out.append(ufunc(*subarray_ins, **kwargs))


        outputs.append(type(inputs[0])(subarrays_out, shape_out, labels_out))

    if len(outputs) == 1:
        return outputs[0]
    else:
        return tuple(outputs)

def apply_ufunc_mat_vec(ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    A function to apply a limited set of ufuncs for BlockMatrix and BlockVector
    """
    # Convert any numpy scalar inputs to floats so that you don't trigger the 
    # __array_ufunc__ interface again
    inputs = [
        float(input) if isinstance(input, Number) else input 
        for input in inputs
    ]

    if ufunc == np.add:
        return operator.add(*inputs)
    elif ufunc == np.subtract:
        return operator.sub(*inputs)
    elif ufunc == np.multiply:
        return operator.mul(*inputs)
    elif ufunc == np.divide:
        return operator.truediv(*inputs)
    elif ufunc == np.power:
        return operator.pow(*inputs)
    else:
        return NotImplemented
