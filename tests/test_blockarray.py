"""
Test the operations in `blockarray.blockarray`
"""

import operator

import pytest
import numpy as np

import blockarray.blockarray as ba

@pytest.fixture()
def setup_barray_a():
    a = np.ones((4, 4))
    b = np.ones((4, 2))
    c = np.ones((2, 4))
    d = np.ones((2, 2))
    return ba.BlockArray([[a, b], [c, d]]), (a, b, c, d)

@pytest.fixture()
def setup_barray_b():
    a = np.ones((4, 4))
    b = np.ones((4, 2))
    c = np.ones((2, 4))
    d = np.ones((2, 2))
    return ba.BlockArray([[a, b], [c, d]]), (a, b, c, d)

class TestBlockArray:

    def test_create_collapsed(self):
        """
        Test that creating reduced/collapsed blocks returns the subarray itself
        """
        a = np.ones((2, 2))
        A = ba.BlockArray([a], shape=(-1, -1))
        assert (A is a)

    def test_index(self, setup_barray_a):
        B, (a, b, c, d) = setup_barray_a
        assert np.all(B.sub[0, 0] == a)
        assert np.all(B.sub[0, 1] == b)
        assert np.all(B.sub[1, 0] == c)
        assert np.all(B.sub[1, 1] == d)

    def test_bshape(self, setup_barray_a):
        A, *_ = setup_barray_a
        assert A.f_bshape == ((4, 2), (4, 2))
        print(f"A.bshape = {A.f_bshape}")
        print(f"A[:, :].bshape = {A[:, :].f_bshape}")
        print(f"A[:, 0].bshape = {A[:, 0].f_bshape}")
        print(f"A[0, :].bshape = {A[0, :].f_bshape}")

    def test_squeeze(self, setup_barray_a):
        A, *_ = setup_barray_a
        dd = A[0, :]
        assert dd.unsqueeze().f_bshape == ((4,), (4, 2))

        dd = A[0:1, :]
        assert dd.squeeze().f_bshape == (4, (4, 2))

class TestMath:

    @pytest.fixture(
        params=[
            operator.add, operator.sub,
            operator.mul, operator.truediv,
            # operator.pow
        ]
    )
    def binary_op(self, request):
        """
        Return a binary operation
        """
        return request.param

    def test_elementwise_binary_op(self, binary_op, setup_barray_a, setup_barray_b):
        """
        Test a generic element-wise binary operation

        This should test whether a operation applied across each block gives the
        same result as an equivalent operation on the BlockArrays
        """
        a, *_ = setup_barray_a
        b, *_ = setup_barray_b

        c_array_result = binary_op(a, b).sub[:].flat
        c_array_reference = tuple(
            binary_op(ai, bi) for ai, bi in zip(a.sub[:].flat, b.sub[:].flat)
        )

        correct_subarrays = [
            np.all(sub_res == sub_ref)
            for sub_res, sub_ref in zip(c_array_result, c_array_reference)
        ]

        assert all(correct_subarrays)

    def test_ufunc(self, setup_barray_a):
        A, *_ = setup_barray_a
        for op in [np.add, np.multiply, np.divide]:
            D = op(5.0, A)
            _D = op(5.0, A.to_mono_ndarray())
            assert np.all(np.isclose(D.to_mono_ndarray(), _D))

        for op in [np.add, np.multiply, np.divide]:
            D = op(np.float64(5.0), A)
            _D = op(np.float64(5.0), A.to_mono_ndarray())
            assert np.all(np.isclose(D.to_mono_ndarray(), _D))

# TODO: This isn't the right way to parameterize a test function
@pytest.fixture(params=[
    (5, 5, (1, 4)),
    ((2, 4), (2, 4), (1, 1)),
    ((2, 4), (2, 4), 1)
])
def setup_bshape(request):
    return request.param

def test_ones(setup_bshape):
    bshape = setup_bshape
    A = ba.zeros(bshape)
    assert A.f_bshape == bshape

def test_zeros(setup_bshape):
    bshape = setup_bshape
    A = ba.zeros(bshape)
    assert A.f_bshape == bshape

def test_rand(setup_bshape):
    bshape = setup_bshape
    A = ba.zeros(bshape)
    assert A.f_bshape == bshape




# def test_to_ndarray():



if __name__ == '__main__':
    test_index()
    test_add()
    test_sub()
    test_mul()
    test_div()
    test_bshape()
    test_squeeze()
    # test_ufunc()
    # test_ones()
    # test_power()