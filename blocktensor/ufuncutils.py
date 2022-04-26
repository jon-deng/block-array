"""
Module implementing `ufunc` logic
"""

import typing

Signature = typing.Tuple[str, ...]

def parse_ufunc_signature(
        sig: str
    ) -> typing.Tuple[typing.Tuple[Signature, ...], Signature]:
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

def interpret_ufunc_signature(sig_inputs, sig_outputs):
    """
    Interprets a ufunc signature
    """
    # TODO: Will have to handle weird signatures where output dimension names
    # do not match and of the input dimension names

    # Get the set of free dimension names and contract (cont) dimension names
    free_names = {name for sig_output in sig_outputs for name in sig_output}
    cnrt_names = {
        name for sig_input in sig_inputs for name in sig_input
        if name not in free_names
    }
    
    # For each free dimension name, record the input number and axis number that
    # it occurs in
    free_name_to_input = {
        name: (ii_input, ii_ax)
        for ii_input, sig_input in enumerate(sig_inputs)
        for ii_ax, name in enumerate(sig_input)
        if name in free_names
    }
    assert set(free_name_to_input.keys()) == free_names

    # For each contracted dimension name, record the axis indices it occurs in 
    # for each input
    cnrt_name_to_input = {name: [] for name in list(cnrt_names)}
    for ii_input, sig_input in enumerate(sig_inputs):
        for ii_ax, name in enumerate(sig_input):
            if name in cnrt_name_to_input:
                cnrt_name_to_input[name].append(tuple([ii_input, ii_ax]))

    return free_name_to_input, cnrt_name_to_input

def split_shapes_by_signatures(shapes, sig_inputs):
    """
    Splits a list of shapes into lists of elementwise dims and core dims
    """
    ewise_shapes = [
        shape[:len(sig)] for shape, sig in zip(shapes, sig_inputs)
    ]
    core_shapes = [
        shape[-len(sig):] for shape, sig in zip(shapes, sig_inputs)
    ]
    return ewise_shapes, core_shapes

def calculate_output_shapes(
    ewise_input_shapes, core_input_shapes, sig_inputs, sig_outputs,
    free_name_to_input=None
    ):
    """
    Calculate the shape of the output BlockArray
    """
    # Check that the element wise dims of all inputs are the same
    # TODO: support broadcasting?
    for shapea, shapeb in zip(ewise_input_shapes[:-1], ewise_input_shapes[1:]):
        assert shapea == shapeb

    if free_name_to_input is None:
        _, free_name_to_input = interpret_ufunc_signature(sig_inputs, sig_outputs)

    ewise_output_shape = ewise_input_shapes[0]
    ewise_output_shapes = [ewise_output_shape] * len(sig_outputs)

    core_output_shapes = [
        tuple([
            core_input_shapes[free_name_to_input[label][0]][free_name_to_input[label][1]] 
            for label in sig
        ]) 
        for sig in sig_outputs
    ]

    return ewise_output_shapes, core_output_shapes
    

    


