"""
Test the operations in blockmath.py
"""

import numpy as np

import blockarray.blockarray as ba


a = np.ones((4, 4))
b = np.ones((4, 2))
c = np.ones((2, 4))
d = np.ones((2, 2))
A = ba.BlockArray([[a, b], [c, d]])

a = 2 * np.ones((4, 4))
b = 2 * np.ones((4, 2))
c = 2 * np.ones((2, 4))
d = 2 * np.ones((2, 2))
B = ba.BlockArray([[a, b], [c, d]])


def _test_elementwise_binary_op(sub_op, a, b, block_op=None):
    """
    Test a generic element-wise binary operation

    This should test whether a operation applied across each block gives the
    same result as an equivalent operation on the BlockArrays
    """
    if block_op is None:
        block_op = sub_op
    c_array_result = block_op(a, b).subarrays_flat
    c_array_reference = tuple([sub_op(ai, bi) for ai, bi in zip(a.subarrays_flat, b.subarrays_flat)])

    correct_subarrays = [
        np.all(sub_res == sub_ref)
        for sub_res, sub_ref in zip(c_array_result, c_array_reference)]

    assert all(correct_subarrays)

def test_add():
    _test_elementwise_binary_op(lambda x, y: x+y, A, B)

def test_sub():
    _test_elementwise_binary_op(lambda x, y: x-y, A, B)

def test_mul():
    _test_elementwise_binary_op(lambda x, y: x*y, A, B)

def test_div():
    _test_elementwise_binary_op(lambda x, y: x/y, A, B)

def test_power():
    _test_elementwise_binary_op(lambda x, y: x**y, A, B, ba.power)

def test_bshape():
    assert A.bshape == ((4, 2), (4, 2))
    print(f"A.bshape = {A.bshape}")
    print(f"A[:, :].bshape = {A[:, :].bshape}")
    print(f"A[:, 0].bshape = {A[:, 0].bshape}")
    print(f"A[0, :].bshape = {A[0, :].bshape}")

# def test_to_ndarray():



if __name__ == '__main__':
    test_add()
    test_sub()
    test_mul()
    test_div()
    test_bshape()
    # test_power()