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
from typing import Tuple, List, Mapping, TypeVar, Union, Callable
import numpy as np

from . import blockarray as ba, blockmat as bm, blockvec as bv, subops
from . import typing

Signature = Tuple[str, ...]
Signatures = List[Signature]

Shapes = List[typing.Shape]

Perm = List[int]

T = TypeVar('T')
Input = Union[ba.BlockArray[T], Number]

V = TypeVar('V')

## Signature processing functions
def parse_ufunc_signature(
        sig_str: str
    ) -> Tuple[Signatures, Signatures]:
    """
    Parse a `ufunc.signature` string into a tuple format

    For a ufunc signature string:
        `'(i,j),(j,k)->(i,k)'`
    this function represents the inputs and output components of the signature
    as:
        `[('i', 'j'), ('j', 'k')], [('i', 'k')]`

    Parameters
    ----------
    sig_str : str
        A `ufunc` signature string

    Returns
    -------
    sig_inputs, sig_outputs :
        A list of tuples representing the input/output signatures
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
        Mapping[str, List[Tuple[int, int]]],
        Mapping[str, List[Tuple[int, int]]]
    ]:
    """
    Interprets a `ufunc` signature

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
    free_dname_to_ins, redu_dname_to_ins: Dict
        A mapping of free/reduced dimension names to inputs and axes.

        This maps the dimension name to a tuple of 2 integers `(nin, dim)`
        containing the input number (`nin`) and core dimension (`dim`)
        that the free label corresponds to.
        For example, a signature '(i,j),(j,k)->(i,k)' has free dimension names
        of 'i,k' and would have
            `free_dname_to_ins = {'i': [(0, 0)], 'k': [(1, 1)]}`
        The same signature has reduced dimension names of 'j' and would have
            `redu_dname_to_ins = {'j': [(0, 1), (1, 0)]}`
    """
    # TODO: Will have to handle weird signatures where output dimension names
    # do not match any of the input dimension names

    # Get the set of free dimension names and contract (cont) dimension names
    free_names = {name for sig_out in sig_outs for name in sig_out}
    redu_names = {
        name for sig_in in sig_ins for name in sig_in
        if name not in free_names
    }

    # For each free/reduced dimension name, record the input number and axis number that
    # it occurs in
    free_dname_to_ins = {name: [] for name in list(free_names)}
    redu_dname_to_ins = {name: [] for name in list(redu_names)}
    for dname_to_ins in [free_dname_to_ins, redu_dname_to_ins]:
        for ii_input, sig_input in enumerate(sig_ins):
            for ii_ax, name in enumerate(sig_input):
                if name in dname_to_ins:
                    dname_to_ins[name].append((ii_input, ii_ax))
    assert set(free_dname_to_ins.keys()) == free_names
    assert set(redu_dname_to_ins.keys()) == redu_names

    return free_dname_to_ins, redu_dname_to_ins

## Output shape/indexing function
def make_gen_in_multi_index(
        std_shape_ins: List[int],
        sig_ins: Signatures,
        sig_out: Signature
    ) -> Callable[[typing.MultiIntIndex], typing.MultiStdIndex]:
    """
    Return a function that generates indices for inputs corresponding to an output index

    Parameters
    ----------
    std_shape_ins :
        A list of input shapes in 'standard' order; all loop dimensions should
        be the first axes followed by all the core dimensions.
    sig_ins :
        Signatures (as returned by `parse_ufunc_signature`) for inputs
    sig_out :
        Signature for the output

    Returns
    -------
    gen_in_multi_index :
        Function that returns indices for each input corresponding to an output
        index.
    """
    free_name_to_output = {label: ii for ii, label in enumerate(sig_out)}

    loop_ndim_ins = [
        len(shape_in)-len(sig_in)
        for shape_in, sig_in in zip(std_shape_ins, sig_ins)
    ]
    def gen_in_multi_index(out_multi_idx):
        """
        Return corresponding input indices for an output index

        A corresponding input index indexes the portion of the input
        involved in computing the specified output index. For example, consider
        the input shapes
            `(1,)` and `(5,)`
        and a `ufunc` with signature
            `'(),()->()'`.
        Then the output shape is broadcast to
            `(5,)`.
        The output subarray at `(4,)` is computed from the input subarrays
        at `(0,)` and `(4,)`, which are the corresponding input
        indices.

        In the example above, all dimensions are loop dimensions; indexing with
        core dimensions is a little trickier. In the current implementation:
            - 'free' core dimensions are treated like loop-dimensions except
            they do not broadcast since core axis sizes must match exactly.
            - 'reduced' core dimensions are indexed with a `:`. That is, all
            subarrays along reduces axes are selected.
        """
        l_midx_outs = out_multi_idx[:len(out_multi_idx)-len(sig_out)]
        c_midx_outs = out_multi_idx[len(out_multi_idx)-len(sig_out):]

        # The `np.minimum` call against `l_shape-1` takes care of broadcasting
        # input axes with size 1, against output axes with size > 1
        # The `[len(l_midx_outs)-n:]` takes care of broadcasting missing input
        # axes against non-missing output axes
        l_shape_ins = [shape[:len(shape)-len(sig)] for shape, sig in zip(std_shape_ins, sig_ins)]
        l_midx_ins = [
            # Convert to int here because indexing doesn't handle np.int types well
            tuple(
                int(ii)
                for ii in np.minimum(
                    l_midx_outs[len(l_midx_outs)-n:],
                    np.array(l_shape)-1
                )
            )
            for n, l_shape in zip(loop_ndim_ins, l_shape_ins)
        ]
        c_midx_ins = [
            tuple(
                c_midx_outs[free_name_to_output[label]]
                if label in free_name_to_output
                else slice(None)
                for label in sig_input
            )
            for sig_input in sig_ins
        ]

        midx_ins = [
            loop+core for loop, core in zip(l_midx_ins, c_midx_ins)
        ]
        return midx_ins

    return gen_in_multi_index

def apply_permutation(arg: Union[List[T], Tuple[T]], perm: Perm) -> Union[List[T], Tuple[T]]:
    """
    Return a permutation of a list

    Parameters
    ----------
    arg : List or Tuple
        The list/tuple to permute
    perm :
        The permutation to apply. This should be a tuple containing integers
        between `0` to `len(arg)-1` ordered according to the desired permutation.

    Returns
    -------
        The permuted input
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

def undo_permutation(arg: Union[List[T], Tuple[T]], perm: Perm) -> Union[List[T], Tuple[T]]:
    """
    Return the result of undoing a permutation on an input

    Parameters
    ----------
    arg : List or Tuple
        The list/tuple to permute
    perm :
        The permutation to undo. This should be a tuple containing integers
        between `0` to `len(arg)-1` ordered according to the desired permutation.

    Returns
    -------
        The un-permuted input
    """
    # create a reverse permutation
    undo_perm = [None]*len(perm)
    for ii, idx in enumerate(perm):
        undo_perm[idx] = ii
    return apply_permutation(arg, tuple(undo_perm))

def conv_neg(n: int, size: int) -> int:
    """
    Convert a negative integer index to the equivalent positive one

    Parameters
    ----------
    n : int
        The index (possible negative)
    size : int
        The size of the axis/array being indexed

    Returns
    -------
    int
        The equivalent positive index
    """
    if n < 0:
        return size+n
    else:
        return n

# Broadcasting functions
def dec_broadcast_none(fun):
    """
    Return a decorated function that also broadcasts over `None`

    This is used with the `broadcast_*` functions.
    """
    def wrapped_fun(a, b):
        if a is None:
            return b
        elif b is None:
            return a
        else:
            return fun(a, b)
    return wrapped_fun

def broadcast_size(a: int, b: int) -> int:
    """
    Broadcast a simple axis size

    Parameters
    ----------
    a, b : int
        Axis sizes being broadcast

    Returns
    -------
    int
        The broadcast axis size
    """
    if a == 1 or a == -1:
        return b
    elif b == 1 or b == -1:
        return a
    elif a == b:
        return a
    else:
        raise ValueError(f"{a} and {b} are not broadcastable")

@dec_broadcast_none
def broadcast_axis_size(a: typing.AxisSize, b: typing.AxisSize) -> typing.AxisSize:
    """
    Broadcast block axis size

    Parameters
    ----------
    a, b: typing.AxisSize
        Nested axis sizes being broadcast

    Returns
    -------
    int
        Broadcasted block axis size
    """
    if isinstance(a, int) and isinstance(b, int):
        return broadcast_size(a, b)
    elif isinstance(a, int) and isinstance(b, tuple):
        return tuple([broadcast_axis_size(a, bb) for bb in b])
    elif isinstance(a, tuple) and isinstance(b, int):
        return tuple([broadcast_axis_size(aa, b) for aa in a])
    elif isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) == 1:
            return tuple([broadcast_axis_size(a[0], bb) for bb in b])
        elif len(b) == 1:
            return tuple([broadcast_axis_size(aa, b[0]) for aa in a])
        elif len(b) == len(a):
            return tuple([broadcast_axis_size(aa, bb) for aa, bb in zip(a, b)])
        else:
            raise ValueError(f"{a} and {b} are not broadcastable")
    else:
        raise ValueError(f"{a} and {b} are not broadcastable")

@dec_broadcast_none
def broadcast_axis_labels(a: typing.Labels, b: typing.Labels) -> typing.Labels:
    """
    Broadcast axis labels

    Parameters
    ----------
    a, b: Tuple[str, ...]
        Axis labels being broadcast

    Returns
    -------
    Tuple[str, ...]
        Broadcasted block axis labels
    """
    if a == ():
        return b
    elif b == ():
        return a
    elif a == b:
        return a
    else:
        raise ValueError(f"{a} and {b} are not broadcastable")

def broadcast(broadcast_op: Callable[[V, V], V], *inputs: Tuple[V, ...]) -> Tuple[V, ...]:
    """
    Broadcast multiple dimension tuples using a specified broadcast operation

    The `broadcast_op` is used to broadcast each dimension/axis of the input
    tuples against each other, similar to broadcasting of numpy shape tuples.

    Parameters
    ----------
    broadcast_op :
        The broadcasting operation to apply along each axis
    inputs :
        Tuples of axis descriptors (size, labels, etc.) to be broadcast.
    """
    rev_inputs = [input[::-1] for input in inputs]
    return tuple([
        functools.reduce(broadcast_op, axis_inputs)
        for axis_inputs in itertools.zip_longest(*rev_inputs, fillvalue=None)
    ])[::-1]

def broadcast_dims(
        broadcast_op: Callable[[V, V], V],
        std_in_dims: Tuple[V, ...],
        sig_ins: Signatures,
        sig_outs: Signatures,
        free_name_to_in: Mapping[str, Tuple[int, int]],
    ):
    """
    Broadcast a set of dimension tuples while accounting for core dimensions

    A dimension tuple is a tuple contaning information about each axis of an
    n-d array. A common example is the `.shape` attribute for `numpy.ndarray`
    which stores the size of each axis as an integer.

    Parameters
    ----------
    broadcast_op: Callable
        An operation which returns the broadcasted result from two axis descriptors
        of the dimension tuple.
    std_in_dims: Tuple
        A dimension tuple describing some property of each axis of the input.
        The dimension tuple must be in standard order; loop dimensions are first
        followed by core dimenions being last.
    sig_ins, sig_outs:
        Input and output signatures (see `parse_ufunc_signature`)
    free_name_to_ins:
        A mapping from free axis labels to associated inputs (see
        `interpret_ufunc_signature`).
    """
    loop_dims = [dims[:len(dims)-len(sig)] for dims, sig in zip(std_in_dims, sig_ins)]
    core_dims = [dims[len(dims)-len(sig):] for dims, sig in zip(std_in_dims, sig_ins)]
    out_loop_dims = broadcast(broadcast_op, *loop_dims)

    # Note `free_name_to_in[label][0][0]` returns the input number
    # Note `free_name_to_in[label][0][1]` returns the free axis idx
    # for the first instance of a free axis input
    out_core_dims = [
        tuple([
            core_dims[free_name_to_in[label][0][0]][free_name_to_in[label][0][1]]
            for label in sig
        ])
        for sig in sig_outs
    ]

    return [out_loop_dims+core_dims for core_dims in out_core_dims]

# These are helper methods for getting `BlockArray` type attributes from both
# `BlockArray` and scalar (float) type objects
def _f_bshape(array: Input[T]) -> typing.BlockShape:
    """
    Return the `f_bshape` attribute for BlockArrays and scalar inputs
    """
    if isinstance(array, Number):
        return ()
    else:
        return array.f_bshape

def _f_shape(array: Input[T]) -> typing.Shape:
    """
    Return the `f_shape` attribute for BlockArrays and scalar inputs
    """
    if isinstance(array, Number):
        return ()
    else:
        return array.f_shape

def _f_labels(array: Input[T]) -> typing.MultiLabels:
    """
    Return the `f_bshape` attribute for BlockArrays and scalar inputs
    """
    if isinstance(array, Number):
        return ()
    else:
        return array.f_labels

def _f_ndim(array: Input[T]) -> int:
    """
    Return the `f_ndim` attribute for BlockArrays and scalar inputs
    """
    if isinstance(array, Number):
        return 0
    else:
        return array.f_ndim

def unsqueeze(array: Input[T]) -> Input[T]:
    """
    Return the unsqueezed `BlockArray` or scalar
    """
    if isinstance(array, Number):
        return array
    else:
        return array.unsqueeze()

# Ufunc routines
def apply_ufunc_array(ufunc: np.ufunc, method: str, *inputs: Input[T], **kwargs):
    """
    Apply a ufunc on BlockArray inputs

    Parameters
    ----------
    ufunc : np.ufunc
        The numpy `ufunc` to apply (see documentation of `np.ufunc`).
    method : str
        The `ufunc` method to apply. This is one of 'reduce', 'accumulate', etc.
        (see documentation of `np.ufunc`).
    inputs : List of BlockArray or scalar
        The inputs to apply the ufunc on
    kwargs :
        Keyword arguments for `np.ufunc` (see documentation of `np.ufunc`).
    """
    ## Validate inputs
    # Check input types
    if not all([
            isinstance(input, (Number, ba.BlockArray))
            for input in inputs
        ]):
        input_types = [type(x) for x in inputs]
        raise TypeError(f"Inputs must be of type `scalar` or `BlockArray`, not {input_types}")

    # Convert any scalar inputs to `numpy` equivalents so that they can be indexed, etc.
    def require_array(x):
        if isinstance(x, Number):
            # The index makes sure the result is a `numpy` scalar, not 0D array
            return np.array(x)[()]
        else:
            return x
    inputs = [require_array(input) for input in inputs]

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
    axes = [(axis,), ()]

    return _apply_op_core(ufunc.reduce, signature, axes, *inputs, **kwargs)

def _apply_ufunc_accumulate(ufunc: np.ufunc, *inputs: Input[T], **kwargs):
    assert len(inputs) == 1

    # The signature for accumulate type calls is always the below
    signature = '(i)->(i)'
    if 'axis' not in kwargs:
        kwargs['axis'] = 0
    axis = kwargs['axis']
    axes = [(axis,), (axis,)]

    return _apply_op_core(ufunc.accumulate, signature, axes, *inputs, **kwargs)

def _apply_ufunc_outer(ufunc: np.ufunc, *inputs: Input[T], **kwargs):
    return NotImplemented

def _apply_op_core(
        ufunc,
        signature: str,
        baxes: List[typing.Shape],
        *inputs: Input[T],
        **kwargs
    ) -> List[Tuple[List[T], typing.Shape, typing.Labels]]:
    """
    Return the result of applying a function of `numpy` subarrays

    Parameters
    ----------
    ufunc :
        The numpy ufunc-like function to apply on the inputs. This includes the
        ufunc methods `ufunc.reduce`, `ufunc.accumulate`, etc...
    signature :
        The signature of the ufunc (see documentation for generalized universal
        functions in numpy).
    baxes :
        The equivalent of the 'axes' keyword argument for `np.ufunc`. `baxes`
        and `kwargs['axes']` must be consistent for the resulting operation to
        make sense as the axes of the blocks and the axes of the subarrays are
        the same.

        This is included as a separate argument since ufunc methods
        (`ufunc.reduce`, etc.) don't have an `axes` keyword argument, although
        they can be modelled in the same way as the direct `ufunc.__call__`
        method. To see how 'axes' is defined in these cases, see
        `_apply_ufunc_reduce`, etc.
    inputs :
        The list of inputs to apply the ufunc on.
    kargs :
        Optional keyword arguments for the ufunc.
    """
    sig_ins, sig_outs = parse_ufunc_signature(signature)
    nout = len(sig_outs)

    # Check the `baxes` and `kwargs['axes']` are consistent
    # TODO: This should also handle the case where 'axis' is supplied
    if 'axes' in kwargs:
        ncore_dims = [len(sig) for sig in sig_ins+sig_outs]
        sub_baxes = [
            tuple([conv_neg(ii, ndim) for ii in axs])
            for axs, ndim in zip(kwargs['axes'], ncore_dims)
        ]
        if sub_baxes != baxes:
            raise ValueError(
                "`ufunc` 'axes' argument is inconsistent with block 'baxes' argument"
            )

    free_name_to_ins, redu_name_to_ins = interpret_ufunc_signature(sig_ins, sig_outs)

    ## Compute a permutation of the `f_shape` from the axes kwargs
    # This permutation shifts core dimensions to the 'standard' location as
    # the final dimensions of the array
    ndim_ins = [_f_ndim(input) for input in inputs]
    core_ndim_ins = [len(sig) for sig in sig_ins]
    loop_ndim_ins = [ndim-core_ndim for ndim, core_ndim in zip(ndim_ins, core_ndim_ins)]
    ndim_outs = [max(loop_ndim_ins)+len(sig) for sig in sig_outs]
    ndims = ndim_ins + ndim_outs

    # Note that `axs` refers to axes of the full shape
    axes = [
        tuple([conv_neg(ii, ndim) for ii in axs])
        for ndim, axs in zip(ndim_ins+ndim_outs, baxes)
    ]

    # Compute the shape permutation from axes
    # This permutes the axis sizes in shape so the core dimensions are at the end
    # and loop dimensions are at the beginning
    # dimensions tuples that are in this format are prefixed by `std_`
    permuts = [
        tuple([ii for ii in range(ndim) if ii not in set(axs)]) + axs
        for axs, ndim in zip(axes, ndims)
    ]
    permut_ins = permuts[:-nout]
    permut_outs = permuts[-nout:]

    ## Determine the output `f_shape` and `f_labels`
    f_shape_ins = [_f_shape(input) for input in inputs]
    std_f_shape_ins = [apply_permutation(x, perm) for x, perm in zip(f_shape_ins, permut_ins)]
    std_f_shape_outs = broadcast_dims(broadcast_axis_size, std_f_shape_ins, sig_ins, sig_outs, free_name_to_ins)

    f_label_ins = [_f_labels(input) for input in inputs]
    std_f_label_ins = [apply_permutation(x, perm) for x, perm in zip(f_label_ins, permut_ins)]
    std_f_labels_outs = broadcast_dims(broadcast_axis_labels, std_f_label_ins, sig_ins, sig_outs, free_name_to_ins)

    ## Check that reduced dimensions have compatible bshapes
    f_bshape_ins = [_f_bshape(input) for input in inputs]
    std_f_bshape_ins = [apply_permutation(x, perm) for x, perm in zip(f_bshape_ins, permut_ins)]
    std_f_bshape_out = broadcast_dims(broadcast_axis_size, std_f_bshape_ins, sig_ins, sig_outs, free_name_to_ins)

    ## Compute the output shape from the input shape and signature
    # the _ prefix means the permuted shape-type tuple with core dimensions at
    # the end

    # Unsqueeze any collapsed axes for the input before applying the op blockwise
    # ; the blockwise loop only works for non-squeezed axes
    inputs = [unsqueeze(input) for input in inputs]
    shape_ins = [input.shape for input in inputs]

    labels_outs = [
        undo_permutation(labels, perm)
        for labels, perm in zip(std_f_labels_outs, permut_outs)
    ]
    shape_outs = [
        undo_permutation(shape, perm)
        for shape, perm in zip(std_f_shape_outs, permut_outs)
    ]

    ## Compute the outputs block wise by looping over inputs
    outputs = []
    for f_shape_out, labels_out, sig_out, perm_out in zip(shape_outs, labels_outs, sig_outs, permut_outs):
        # Unsqueeze the output shape as well
        shape_out = ba.unsqueeze_shape(f_shape_out)
        subarrays_out = _apply_op_blockwise(
            ufunc, inputs, shape_ins, shape_out, sig_ins, sig_out, permut_ins, perm_out, op_kwargs=kwargs)
        outputs.append((subarrays_out, f_shape_out, labels_out))

    return outputs

def _apply_op_blockwise(
        op,
        inputs: List[Input[T]],
        shape_ins: Shapes,
        shape_out: typing.Shape,
        sig_ins: Signatures,
        sig_out: Signatures,
        permut_ins: List[Perm],
        perm_out: Perm,
        op_kwargs=None
    ) -> List[T]:
    """
    Return the subarrays from applying an operation over blocks of `BlockArray`s

    This roughly works as follow:
        - Output subarrays along loop dimensions result from applying `ufunc`
        blockwise along corresponding loop dimensions on inputs.
        - Output subarrays along core free dimensions result from applying `ufunc`
        elementwise along corresponding core free dimensions on inputs. That is
        core free dimensions are treated like loop dimensions.
        - Output subarrays along reduced dimensions are not present, since these
        dimensions are collapsed/reduced. To perform the collapsed/reducing
        operation for each subarray along elementwise blocks, subarrays are
        concatenated along the reduced dimensions and ufuncs are then applied on
        the single concatenated subarray.
    """
    # `shape_ins` must be in standard order with core dimensions at the end
    # since this is how `make_gen_in_multi_index` works
    std_shape_ins = [apply_permutation(shape, perm) for shape, perm in zip(shape_ins, permut_ins)]
    gen_in_midx = make_gen_in_multi_index(std_shape_ins, sig_ins, sig_out)

    subarrays_out = []

    def _apply_output_op(inputs, midx_out, permut_ins, perm_out, **op_kwargs):
        std_midx_out = apply_permutation(midx_out, perm_out)
        std_midx_ins = gen_in_midx(std_midx_out)
        midx_ins = [
            undo_permutation(idx, perm)
            for idx, perm in zip(std_midx_ins, permut_ins)
        ]
        subarray_ins = [
            input[midx_in] for input, midx_in in zip(inputs, midx_ins)
        ]
        subarray_ins = [
            subarray.to_mono_ndarray()
            if isinstance(subarray, ba.BlockArray)
            else subops.unwrap(subarray)
            for subarray in subarray_ins
        ]

        return op(*subarray_ins, **op_kwargs)

    if shape_out == ():
        subarrays_out = [
            _apply_output_op(inputs, (), permut_ins, perm_out, **op_kwargs)
        ]
    else:
        subarrays_out = [
            _apply_output_op(inputs, midx_out, permut_ins, perm_out, **op_kwargs)
            for midx_out in itertools.product(
                *[range(ax_size) for ax_size in shape_out]
            )
        ]
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
