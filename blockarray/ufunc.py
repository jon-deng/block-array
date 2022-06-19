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
import functools
from typing import Tuple, List, Mapping, Optional, TypeVar, Union
import numpy as np

from . import blockarray as ba, blockmat as bm, blockvec as bv
from . import typing

Signature = Tuple[str, ...]
Signatures = List[Signature]

Shapes = List[typing.Shape]

Perm = List[int]

T = TypeVar('T')
Input = Union[ba.BlockArray[T], Number]

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
    Split a list of shapes into loop and core shapes
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
    Calculate and output shape from input shapes and a signature
    """
    # Check that the element wise dims of all inputs are the same
    # TODO: support broadcasting?
    loop_shape_out = np.broadcast_shapes(*loop_shape_ins)
    # for shapea, shapeb in zip(loop_shape_ins[:-1], loop_shape_ins[1:]):
    #     assert shapea == shapeb

    if free_name_to_in is None:
        free_name_to_in, _ = interpret_ufunc_signature(sig_ins, sig_outs)

    # loop_shape_out = loop_shape_ins[0]
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
        shape_ins: List[int],
        sig_ins: Signatures,
        sig_out: Signature
    ):
    """
    Make a function that generates indices for inputs given an output index
    """
    free_name_to_output = {label: ii for ii, label in enumerate(sig_out)}

    loop_ndim_ins = [
        len(shape_in)-len(sig_in)
        for shape_in, sig_in in zip(shape_ins, sig_ins)
    ]
    def gen_in_multi_index(out_multi_idx):
        if len(sig_out) == 0:
            l_midx_outs = out_multi_idx
            c_midx_outs = ()
        else:
            l_midx_outs = out_multi_idx[:-len(sig_out)]
            c_midx_outs = out_multi_idx[-len(sig_out):]

        l_midx_ins = [
            l_midx_outs[-n:] if n != 0 else ()
            for n in loop_ndim_ins
        ]
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

def apply_permutation(arg: List[T], perm: Perm) -> List[T]:
    """
    Return a permutation of a list
    """
    # check the list to permute is valid
    assert len(arg) == len(perm)

    # Check if permuting an empty argument
    if len(arg) == 0:
        return arg
    else:
        # check the permutation is valid
        if max(perm) != len(perm) - 1:
            raise ValueError(f"The permutation {perm} is not valid for an array of size {len(arg)}")

        return type(arg)([arg[ii] for ii in perm])

def conv_neg(n: int, size: int) -> int:
    """
    Convert a negative integer index to the equivalent positive one
    """
    if n < 0:
        return size+n
    else:
        return n

# Broadcasting functions
def broadcast_size(a, b):
    if a is None:
        return b
    elif b is None:
        return a
    elif a == 1 and b > 1:
        return b
    elif a > 1 and b == 1:
        return a
    elif a == b:
        return a
    else:
        raise ValueError(f"{a} and {b} are not broadcastable")

def broadcast_label(a, b):
    if a is None:
        return b
    elif b is None:
        return a
    elif a == b:
        return a
    else:
        raise ValueError(f"{a} and {b} are not broadcastable")

def rbroadcast(broadcast_op, a, b):
    if isinstance(a, tuple) or isinstance(b, tuple):
        if not isinstance(a, tuple):
            _a = tuple([a])
        else:
            _a = a

        if not isinstance(b, tuple):
            _b = tuple([b])
        else:
            _b = b

        if len(_a) == 1 and len(_b) >= 1:
            return tuple([rbroadcast(broadcast_op, _a[0], bb) for bb in _b])
        elif len(_a) >= 1 and len(_b) == 1:
            return tuple([rbroadcast(broadcast_op, aa, _b[0]) for aa in _a])
        elif len(_a) == len(_b):
            return tuple([rbroadcast(broadcast_op, aa, bb) for aa, bb in zip(_a, _b)])
        else:
            raise ValueError(f"{a} and {b} are not broadcastable due to different lengths")
    else:
        return broadcast_op(a, b)

def broadcast(broadcast_op, *inputs):
    rev_inputs = [input[::-1] for input in inputs]
    return tuple([
        functools.reduce(
            functools.partial(rbroadcast, broadcast_op),
            axis_inputs
        )
        for axis_inputs in itertools.zip_longest(*rev_inputs, fillvalue=None)
    ])[::-1]

def broadcast_dims(
        broadcast_op,
        input_dims,
        sig_ins,
        sig_outs,
        permut_ins,
        free_name_to_in,
    ):
    """
    Broadcast a set of dimensions to an output dimension

    The dimension is a tuple descriptors describing properties of each axis
    along the n-dimensional array.
    A common example is the `.shape` attribute for `numpy.ndarray` which stores
    the size of each axis as an integer.
    """
    _in_dims = [
        apply_permutation(dims, perm) for dims, perm in zip(input_dims, permut_ins)
    ]
    loop_dims = [dims[:len(dims)-len(sig)] for dims, sig in zip(_in_dims, sig_ins)]
    core_dims = [dims[len(dims)-len(sig):] for dims, sig in zip(_in_dims, sig_ins)]
    out_loop_dims = broadcast(broadcast_op, *loop_dims)

    out_core_dims = [
        tuple([
            core_dims[free_name_to_in[label][0]][free_name_to_in[label][1]]
            for label in sig
        ])
        for sig in sig_outs
    ]

    return [out_loop_dims+core_dims for core_dims in out_core_dims]

# Helper method for getting `BlockArray` attributes from scalars
def _f_bshape(array: Input[T]) -> typing.BlockShape:
    """
    Return the `f_bshape` attribute for BlockArrays and scalar inputs
    """
    if isinstance(array, Number):
        return ()
    else:
        return array.f_bshape

def _f_shape(array):
    """
    Return the `f_shape` attribute for BlockArrays and scalar inputs
    """
    if isinstance(array, Number):
        return ()
    else:
        return array.f_shape

def _f_labels(array):
    """
    Return the `f_bshape` attribute for BlockArrays and scalar inputs
    """
    if isinstance(array, Number):
        return ()
    else:
        return array.f_labels

def _f_ndim(array: Input[T]) -> typing.BlockShape:
    """
    Return the `f_ndim` attribute for BlockArrays and scalar inputs
    """
    if isinstance(array, Number):
        return 0
    else:
        return array.f_ndim

def unsqueeze(array):
    """
    Return the unsqueeze `BlockArray` or scalar
    """
    if isinstance(array, Number):
        return array
    else:
        return array.unsqueeze()

# Ufunc routines
def apply_ufunc_array(ufunc: np.ufunc, method: str, *inputs: Input[T], **kwargs):
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
        outputs = _apply_ufunc_call(ufunc, *inputs, **kwargs)
    elif method == 'reduce':
        outputs = _apply_ufunc_reduce(ufunc, *inputs, **kwargs)
    elif method == 'outer':
        outputs = _apply_ufunc_outer(ufunc, *inputs, **kwargs)
    elif method == 'accumulate':
        outputs = _apply_ufunc_accumulate(ufunc, *inputs, **kwargs)
    else:
        return NotImplemented

    # In the first case a single output tuple of subarrays, shape and labels
    # is returned
    if len(outputs) == 1:
        subarrays_out, shape_out, labels_out = outputs[0]
        return ba.BlockArray(subarrays_out, shape_out, labels_out)
    else:
        return [
            ba.BlockArray(subarrays_out, shape_out, labels_out)
            for subarrays_out, shape_out, labels_out in outputs
        ]

def _apply_ufunc_call(ufunc: np.ufunc, *inputs: Input[T], **kwargs):
    """
    Apply a ufunc on a sequence of BlockArray inputs with `__call__`

    Parameters
    ----------
    ufunc: np.ufunc
        A numpy `ufunc` to apply
    inputs:
        A list of inputs to apply the `ufunc` on
    kwargs:
        keyword arguments to supply to the ufunc. These are documented in
        https://numpy.org/doc/stable/reference/ufuncs.html#optional-keyword-arguments
    """
    ## Parse signature into nice/standard format
    if ufunc.signature is None:
        signature = ','.join(['()']*ufunc.nin) + '->' + ','.join(['()']*ufunc.nout)
    else:
        signature = ufunc.signature

    sig_ins, sig_outs = parse_ufunc_signature(signature)

    if 'axes' in kwargs:
        axes = kwargs['axes']
    else:
        axes = [
            tuple([-ii for ii in range(len(sig), 0, -1)])
            for sig in sig_ins+sig_outs
        ]

    return _apply_op_core(ufunc, signature, axes, *inputs, **kwargs)

def _apply_ufunc_reduce(ufunc: np.ufunc, *inputs: Input[T], **kwargs):
    assert len(inputs) == 1

    # The signature for reduce type calls is always the below
    signature = '(i)->()'

    if 'axis' not in kwargs:
        kwargs['axis'] = 0
    axis = kwargs['axis']
    axes = [(axis,), (axis,), ()]

    return _apply_op_core(ufunc.reduce, signature, axes, *inputs, **kwargs)

def _apply_ufunc_accumulate(ufunc: np.ufunc, *inputs: Input[T], **kwargs):
    assert len(inputs) == 1

    # The signature for accumulate type calls is always the below
    signature = '(i)->(i)'
    if 'axis' not in kwargs:
        kwargs['axis'] = 0
    axis = kwargs['axis']
    axes = [(axis,), (axis,), ()]

    return _apply_op_core(ufunc.accumulate, signature, axes, *inputs, **kwargs)

def _apply_ufunc_outer(ufunc: np.ufunc, *inputs: Input[T], **kwargs):
    return NotImplemented

def _apply_op_core(
        ufunc,
        signature: str,
        baxes: typing.Shape,
        *inputs: Input[T],
        **kwargs
    ) -> List[Tuple[List[T], typing.Shape, typing.Labels]]:
    sig_ins, sig_outs = parse_ufunc_signature(signature)
    nout = len(sig_outs)

    free_name_to_in, redu_name_to_in = interpret_ufunc_signature(sig_ins, sig_outs)

    ## Compute a permutation of the `f_shape` from the axes kwargs
    # This permutation shifts core dimensions to the 'standard' location as
    # the final dimensions of the array
    ndim_ins = [_f_ndim(input) for input in inputs]
    _loop_ndim_ins = [ndim-len(sig) for ndim, sig in zip(ndim_ins, sig_ins)]
    ndim_outs = [max(_loop_ndim_ins)+len(sig) for sig in sig_outs]
    ndims = ndim_ins + ndim_outs

    # TODO: be careful to convert `axs` to full axes arguments
    axes = [
        tuple([conv_neg(ii, ndim) for ii in axs])
        for ndim, axs in zip(ndim_ins+ndim_outs, baxes)
    ]

    # Compute the shape permutation from axes
    # This permutes the axis sizes in shape so the core dimensions are at the end
    # and elementwise dimensions are at the beginning
    permuts = [
        tuple([ii for ii in range(ndim) if ii not in set(axs)]) + axs
        for axs, ndim in zip(axes, ndims)
    ]
    permut_ins = permuts[:-nout]
    permut_outs = permuts[-nout:]

    ## Determine the output `f_shape` and `f_labels`
    f_shape_ins = [_f_shape(input) for input in inputs]
    _f_shape_outs = broadcast_dims(broadcast_size, f_shape_ins, sig_ins, sig_outs, permut_ins, free_name_to_in)

    f_label_ins = [_f_labels(input) for input in inputs]
    _f_labels_outs = broadcast_dims(broadcast_label, f_label_ins, sig_ins, sig_outs, permut_ins, free_name_to_in)

    ## Check that reduced dimensions have compatible bshapes
    _bshape_ins = [
        apply_permutation(_f_bshape(input), perm)
        for input, perm in zip(inputs, permut_ins)
    ]
    for redu_dim_name, redu_dim_info in redu_name_to_in.items():
        redu_bshapes = [_bshape_ins[ii_in][ii_dim] for ii_in, ii_dim in redu_dim_info]
        if not (redu_bshapes[:-1] == redu_bshapes[1:]):
            raise ValueError(
                f"Core dimension {redu_dim_name} has incompatible block shapes"
                f"of {redu_bshapes}."
            )

    ## Compute the output shape from the input shape and signature
    # the _ prefix means the permuted shape-type tuple with core dimensions at
    # the end

    # Unsqueeze any collapsed axes for the input before applying the op blockwise
    # ; the blockwise loop only works for non-squeezed axes
    inputs = [unsqueeze(input) for input in inputs]
    shape_ins = [input.shape for input in inputs]

    labels_outs = [
        apply_permutation(labels, perm)
        for labels, perm in zip(_f_labels_outs, permut_outs)
    ]
    shape_outs = [
        apply_permutation(shape, perm)
        for shape, perm in zip(_f_shape_outs, permut_outs)
    ]

    ## Compute the outputs block wise by looping over inputs
    outputs = []
    for f_shape_out, labels_out, sig_out, perm_out in zip(shape_outs, labels_outs, sig_outs, permut_outs):
        # Unsqueeze the output shape as well
        _shape_out = tuple(ax_size if ax_size != -1 else 1 for ax_size in f_shape_out)
        subarrays_out = _apply_op_blockwise(
            ufunc, inputs, shape_ins, sig_ins, sig_out, _shape_out, perm_out, permut_ins, op_kwargs=kwargs)
        outputs.append((subarrays_out, f_shape_out, labels_out))

    return outputs

def _apply_op_blockwise(
        op,
        inputs: List[Input[T]],
        shape_ins: Shapes,
        sig_ins: Signatures,
        sig_out: Signatures,
        shape_out: typing.Shape,
        perm_out: Perm,
        permut_ins: List[Perm],
        op_kwargs=None
    ) -> List[T]:
    """
    Return the subarrays from applying an operation over blocks of `BlockArray`s
    """
    ## Remove any scalar inputs from the list of inputs/signatures
    # These should be put back in when the ufunc is computed on subarrays
    # scalar_descr_inputs = [(ii, x) for ii, x in enumerate(inputs) if isinstance(x, Number)]
    # inputs = [x for x in inputs if not isinstance(x, Number)]
    # sig_ins = [sig for x, sig in zip(inputs, sig_ins) if not isinstance(x, Number)]
    # inputs = [x for x in inputs if not isinstance(x, Number)]
    sig_ins = [sig for x, sig in zip(inputs, sig_ins)]

    gen_in_midx = make_gen_in_multi_index(shape_ins, sig_ins, sig_out)

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
        # for ii, scalar in scalar_descr_inputs:
        #     subarray_ins.insert(ii, scalar)

        subarrays_out.append(op(*subarray_ins, **op_kwargs))
    return subarrays_out


V = Union[Union[bm.BlockMatrix[T], Number], Union[bv.BlockVector[T], Number]]
def apply_ufunc_mat_vec(
        ufunc: np.ufunc,
        method: str,
        *inputs: V[T],
        **op_kwargs
    ) -> List[V[T]]:
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
