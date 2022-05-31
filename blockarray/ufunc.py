"""
Module implementing `numpy.ufunc` logic


To explain how `numpy.ufunc` logic is extended to block arrays, note that each 
dimension of an nd-array is classified into two types (see np.ufunc 
documentation for more details):
    loop :
        these are dimensions over which a ufunc is applied elementwise
    core :
        these are dimensions of 'core' array that a ufunc operates on

The core dimensions are also associated with a signature which explains the 
relation between core shapes of the inputs/outputs. To extend the ufunc 
logic to block arrays, further divide core dimensions into 
    reduced : 
        dimensions with labels that only appear in input signatures.
        These dimensions 'dissapear' from the output shape
    free : 
        dimensions with labels that appear in both input and output signatures

Applying a ufunc on block arrays, applies the ufunc on each block over all 
loop dimensions and all free dimensions. As a result, the ufunc is applied on
block arrays containing only the reduced dimensions.
"""
import operator
from numbers import Number
import itertools
from typing import Tuple, List, Mapping, Optional, TypeVar
import numpy as np

from . import blockarray as ba
from . import typing

Signature = Tuple[str, ...]
Signatures = List[Signature]

Shapes = List[typing.Shape]
T = TypeVar('T')

def parse_ufunc_signature(
        sig_str: str
    ) -> Tuple[Signatures, Signatures]:
    """
    Parse a ufunc signature into a nicer format

    For a ufunc signature string
    '(i,j),(j,k)->(i,k)'
    this function represents the inputs and output axis labels in a tuple
    `('i', 'j') ('j', 'k') ('i', 'k')`
    """
    # split into input and output signatures
    sig_str = sig_str.replace(' ', '')
    sig_str_inputs, sig_str_outputs = sig_str.split('->')

    # further split the input/output signatures into signatures for each
    # input/output
    sig_inputs = sig_str_inputs.split('),(')
    sig_inputs[0] = sig_inputs[0].replace('(', '')
    sig_inputs[-1] = sig_inputs[-1].replace(')', '')

    sig_outputs = sig_str_outputs.split('),(')
    sig_outputs[0] = sig_outputs[0].replace('(', '')
    sig_outputs[-1] = sig_outputs[-1].replace(')', '')

    # Change the signatures into tuples of symbols
    sig_inputs = [
        tuple() if sig_input == '' else tuple(sig_input.split(','))
        for sig_input in sig_inputs
    ]
    sig_outputs = [
        tuple() if sig_output == '' else tuple(sig_output.split(','))
        for sig_output in sig_outputs
    ]
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

    This returns dictionaries containing information on the 'free' and
    'reduced' axes/dimensions. 'free' dimensions correspond to labels that occur
    in output (and usually inputs too) signatures while 'reduced' dimensions
    correspond to labels that occur only in the inputs.

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
        shapes: typing.Shape,
        sigs: Signatures
    ) -> Tuple[Shapes, Shapes]:
    """
    Splits a list of shapes into lists of elementwise dims and core dims
    """
    loop_shapes = [
        shape[:-len(sig)] if len(sig) != 0 else shape[:]
        for shape, sig in zip(shapes, sigs)
    ]
    core_shapes = [
        shape[-len(sig):] if len(sig) != 0 else ()
        for shape, sig in zip(shapes, sigs)
    ]
    return loop_shapes, core_shapes

def calculate_output_shapes(
        loop_shape_ins: Shapes,
        core_shape_ins: Shapes,
        sig_ins: Signatures,
        sig_outs: Signatures,
        free_name_to_in : Optional[Mapping[str, Tuple[int, int]]]=None
    ) -> Tuple[Shapes, Shapes]:
    """
    Calculate the shape of the output BlockArray
    """
    # Check that the element wise dims of all inputs are the same
    # TODO: support broadcasting?
    for shapea, shapeb in zip(loop_shape_ins[:-1], loop_shape_ins[1:]):
        assert shapea == shapeb

    if free_name_to_in is None:
        free_name_to_in, _ = interpret_ufunc_signature(sig_ins, sig_outs)

    loop_shape_out = loop_shape_ins[0]
    loop_shape_outs = [loop_shape_out] * len(sig_outs)

    core_shape_outs = [
        tuple([
            core_shape_ins[free_name_to_in[label][0]][free_name_to_in[label][1]]
            for label in sig
        ])
        for sig in sig_outs
    ]

    return loop_shape_outs, core_shape_outs

def make_gen_in_multi_index(
        loop_ndim_ins: List[int],
        sig_ins: Signatures,
        sig_out: Signature
    ):
    """
    Make a function that generates indices for inputs given an output index
    """
    free_name_to_output = {label: ii for ii, label in enumerate(sig_out)}

    def gen_in_multi_index(out_multi_idx):
        if len(sig_out) == 0:
            l_midx_outs = out_multi_idx
            c_midx_outs = ()
        else:
            l_midx_outs = out_multi_idx[:-len(sig_out)]
            c_midx_outs = out_multi_idx[-len(sig_out):]

        l_midx_ins = [l_midx_outs[-n:] for n in loop_ndim_ins]
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
            loop+core for loop, core in zip(l_midx_ins, c_midx_ins)
        ]
        return midx_ins

    return gen_in_multi_index

def apply_permutation(arg: List[T], perm: List[int]) -> List[T]:
    """
    Return a permutation of a list
    """
    # check the permutation is valid
    assert max(perm) == len(perm) - 1

    # check the list to permute is valid
    assert len(arg) == len(perm)

    return type(arg)([arg[ii] for ii in perm])

def conv_neg(n, size):
    """
    Convert a negative integer index to the equivalent positive one
    """
    if n < 0:
        return size+n
    else:
        return n


def apply_ufunc_array(ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    Apply a ufunc on sequence of BlockArray inputs
    """
    ## Validate inputs
    # Check input types
    if not all([
            isinstance(input, (Number, ba.BlockArray))
            for input in inputs
        ]):
        input_types = [type(x) for x in inputs]
        raise TypeError(f"Inputs must be of type `scalar` or `BlockArray`, not {input_types}")

    if method == '__call__':
        return _apply_ufunc_call(ufunc, *inputs, **kwargs)
    elif method == 'reduce':
        return _apply_ufunc_reduce(ufunc, *inputs, **kwargs)
    elif method == 'outer':
        return _apply_ufunc_outer(ufunc, *inputs, **kwargs)
    elif method == 'accumulate':
        return _apply_ufunc_accumulate(ufunc, *inputs, **kwargs)
    else:
        return NotImplemented

def _apply_ufunc_call(ufunc: np.ufunc, *inputs, **kwargs):
    """
    Apply a ufunc on a sequence of BlockArray inputs with `__call__`
    """
    ## Validate inputs
    # Check input types
    if not all([
            isinstance(input, (Number, ba.BlockArray))
            for input in inputs
        ]):
        input_types = [type(x) for x in inputs]
        raise TypeError(f"Inputs must be of type `scalar` or `BlockArray`, not {input_types}")

    ## Parse signature into nice/standard format
    if ufunc.signature is None:
        signature = ','.join(['()']*ufunc.nin) + '->' + ','.join(['()']*ufunc.nout)
    else:
        signature = ufunc.signature
    sig_ins, sig_outs = parse_ufunc_signature(signature)

    ## Remove any scalar inputs from the list of inputs/signatures
    # These should be put back in when the ufunc is computed on subarrays
    scalar_descr_inputs = [(ii, x) for ii, x in enumerate(inputs) if isinstance(x, Number)]
    inputs = [x for x in inputs if not isinstance(x, Number)]
    sig_ins = [sig for x, sig in zip(inputs, sig_ins) if not isinstance(x, Number)]

    ## Compute a permutation of the shape from the axes kwargs
    # This permutation shifts core dimensions to the 'standard' location as
    # the final dimensions of the array
    ndim_ins = [input.ndim for input in inputs]
    _loop_ndim_ins = [ndim-len(sig) for ndim, sig in zip(ndim_ins, sig_ins)]
    ndim_outs = [max(_loop_ndim_ins)+len(sig) for sig in sig_outs]
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
    # Remove any axes for scalar inputs
    axes_ins = axes[:-ufunc.nout]
    axes_outs = axes[-ufunc.nout:]
    axes_ins = [axs for x, axs in zip(inputs, axes_ins) if not isinstance(x, Number)]
    axes = axes_ins + axes_outs

    # Compute the shape permutation from axes
    # This permutes the axis sizes in shape so the core dimensions are at the end
    # and elementwise dimensions are at the beginning
    permuts = [
        tuple([ii for ii in range(ndim) if ii not in set(axs)]) + axs
        for axs, ndim in zip(axes, ndims)
    ]
    permut_ins = permuts[:-ufunc.nout]
    permut_outs = permuts[-ufunc.nout:]

    ## Interpret the ufunc signature in order to compute the shape of the output
    free_name_to_in, redu_name_to_in = interpret_ufunc_signature(sig_ins, sig_outs)

    # Check if reduced dimensions have compatible bshapes

    ## Compute the output shape from the input shape and signature
    # the _ prefix means the permuted shape-type tuple with core dimensions at
    # the end
    _shape_ins = [
        apply_permutation(input.shape, perm)
        for input, perm in zip(inputs, permut_ins)
    ]

    # Check that reduced dimensions have compatible bshapes
    _bshape_ins = [apply_permutation(input.bshape, perm) for input, perm in zip(inputs, permut_ins)]
    for redu_dim_name, redu_dim_info in redu_name_to_in.items():
        redu_bshapes = [_bshape_ins[ii_in][ii_dim] for ii_in, ii_dim in redu_dim_info]
        if not (redu_bshapes[:-1] == redu_bshapes[1:]):
            raise ValueError(
                f"Core dimension {redu_dim_name} has incompatible block shapes"
                f"of {redu_bshapes}."
            )

    _labels_ins = [
        apply_permutation(input.labels, perm)
        for input, perm in zip(inputs, permut_ins)
    ]
    _loops_shape_ins, _core_shape_ins = split_shapes_by_signatures(_shape_ins, sig_ins)
    loop_ndim_ins = [len(loop_shape) for loop_shape in _loops_shape_ins]

    _loop_shape_outs, _core_shape_outs = calculate_output_shapes(
        _loops_shape_ins, _core_shape_ins, sig_ins, sig_outs, free_name_to_in
    )

    loop_labels_ins = [
        _labels[:-len(sig_in)] if len(sig_in) != 0 else _labels
        for _labels, sig_in in zip(_labels_ins, sig_ins)
    ]
    core_labels_ins = [
        _labels[-len(sig_in):] if len(sig_in) != 0 else ()
        for _labels, sig_in in zip(_labels_ins, sig_ins)
    ]
    core_labels_outs = [
        tuple([
            core_labels_ins[free_name_to_in[name][0]][free_name_to_in[name][1]]
            for name in sig_out
        ])
        for sig_out in sig_outs
    ]

    _labels_outs = [loop_labels_ins[0] + clabels_out for clabels_out in core_labels_outs]
    _shape_outs = [eshape+cshape for eshape, cshape in zip(_loop_shape_outs, _core_shape_outs)]

    labels_outs = [
        apply_permutation(labels, perm)
        for labels, perm in zip(_labels_outs, permut_outs)
    ]
    shape_outs = [
        apply_permutation(shape, perm)
        for shape, perm in zip(_shape_outs, permut_outs)
    ]
    assert len(shape_outs) == len(sig_outs)

    ## Compute the outputs block wise by looping over inputs
    outputs = []
    for shape_out, labels_out, sig_out, perm_out in zip(shape_outs, labels_outs, sig_outs, permut_outs):
        gen_in_midx = make_gen_in_multi_index(loop_ndim_ins, sig_ins, sig_out)

        subarrays_out = []
        for midx_out in itertools.product(
                *[range(ax_size) for ax_size in shape_out]
            ):
            _midx_out = apply_permutation(midx_out, perm_out)
            _midx_ins = gen_in_midx(_midx_out)
            midx_ins = [
                apply_permutation(_idx, perm)
                for _idx, perm in zip(_midx_ins, permut_ins)
            ]
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

def _apply_ufunc_reduce(ufunc: np.ufunc, *inputs, **kwargs):
    return NotImplemented

def _apply_ufunc_accumulate(ufunc: np.ufunc, *inputs, **kwargs):
    return NotImplemented

def _apply_ufunc_outer(ufunc: np.ufunc, *inputs, **kwargs):
    return NotImplemented


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
