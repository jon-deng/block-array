"""
Basic math for block tensors
"""

from typing import Callable, TypeVar
import functools

from tensor import BlockTensor

T = TypeVar('T')

def validate_elementwise_binary_op(a, b):
    """
    Validates if BlockTensor inputs are applicable
    """
    pass

def _elementwise_binary_op(op: Callable[T, T], a: BlockTensor, b: BlockTensor):
    """
    Compute elementwise binary operation on BlockTensors

    Parameters
    ----------
    op: function
        A function with signature func(a, b) -> c, where a, b, c are vector of 
        the same shape
    a, b: BlockTensor
    """
    array = tuple([op(ai, bi) for ai, bi in zip(a.array, b.array)])
    return type(a)(array, a.labels)

add = functools.partial(_elementwise_binary_op, lambda a, b: a+b)

sub = functools.partial(_elementwise_binary_op, lambda a, b: a-b)

mul = functools.partial(_elementwise_binary_op, lambda a, b: a*b)

div = functools.partial(_elementwise_binary_op, lambda a, b: a/b)

power = functools.partial(_elementwise_binary_op, lambda a, b: a**b)


def _elementwise_unary_op(op: Callable, a: BlockTensor):
    """
    Compute elementwise unary operation on a BlockTensor

    Parameters
    ----------
    a: BlockTensor
    """
    array = tuple([op(ai) for ai in a.array])
    return type(a)(array, a.labels)

neg = functools.partial(_elementwise_unary_op, lambda a: -a)

pos = functools.partial(_elementwise_unary_op, lambda a: +a)

