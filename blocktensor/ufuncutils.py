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
    sig_inputs, sig_output = sig.split('->')

    # further split the inputs signatures into signatures for each input
    sig_inputs = sig_inputs.split('),(') 
    sig_inputs[0] = sig_inputs[0].replace('(', '')
    sig_inputs[-1] = sig_inputs[-1].replace(')', '')

    # Clean extra parentheses from output signature
    sig_output = sig_output.replace('(', '').replace(')', '')

    # Change the signatures into tuples of symbols
    sig_inputs = [tuple(sig_input.split(',')) for sig_input in sig_inputs]
    sig_output = tuple(sig_output.split(','))
    return sig_inputs, sig_output