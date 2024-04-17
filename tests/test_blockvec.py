"""
Test `blockarray.blockvec`
"""

from xml.etree.ElementPath import xpath_tokenizer, xpath_tokenizer_re
import pytest
import operator

import numpy as np
import petsc4py.PETSc as PETSc

from blockarray import blockvec as bvec
from blockarray import blockarray as btensor

# pylint: disable=unused-import
# pylint: disable=missing-function-docstring


@pytest.fixture(params=['numpy', 'petsc'])
def setup_vec(request):
    if request.param == 'numpy':
        a = np.arange(2) + 1
        b = np.arange(3) + 1
        c = np.arange(4) + 1

        vec = bvec.BlockVector((a, b, c), labels=(('a', 'b', 'c'),))
        subvecs = (a, b, c)
        bshape = ((2, 3, 4),)
    else:
        a = PETSc.Vec().createSeq(2)
        b = PETSc.Vec().createSeq(3)
        c = PETSc.Vec().createSeq(4)

        a.array[:] = 2
        b.array[:] = 3
        c.array[:] = 4
        a.assemble()
        b.assemble()
        c.assemble()

        subvecs = (a, b, c)
        bshape = ((2, 3, 4),)
        vec = bvec.BlockVector((a, b, c), labels=(('a', 'b', 'c'),))
    return vec, subvecs, bshape


@pytest.fixture()
def setup_vec_pair(setup_vec):
    veca, *_ = setup_vec
    vecb = veca.copy()
    return veca, vecb


def test_size_shape(setup_vec):
    vec, subvecs, bshape = setup_vec
    assert vec.f_bshape == bshape


def _test_binary_op(op, vec_pair, element_op=None):
    """
    Tests a binary operation against the equivalent operation on the subtensors
    """
    vec_a, vec_b = vec_pair
    element_op = op if element_op is None else element_op
    vec_c = op(vec_a, vec_b)
    for subvec_c, subvec_a, subvec_b in zip(vec_c.sub[:], vec_a.sub[:], vec_b.sub[:]):
        assert (
            np.power(np.subtract(subvec_c, element_op(subvec_a, subvec_b)), 2).sum()
            == 0
        )


def test_add(setup_vec_pair):
    _test_binary_op(operator.add, setup_vec_pair)


def test_div(setup_vec_pair):
    _test_binary_op(operator.truediv, setup_vec_pair)


def test_mul(setup_vec_pair):
    _test_binary_op(operator.mul, setup_vec_pair)


# def test_power(setup_vec_pair):
#     _test_binary_op(operator.pow, setup_vec_pair)


def _test_unary_op(op, vec, element_op=None):
    """
    Tests a binary operation against the equivalent operation on the subtensors
    """
    element_op = op if element_op is None else element_op
    vec_c = op(vec)
    for subvec_c, subvec_a in zip(vec_c.sub[:], vec.sub[:]):
        assert np.power(np.subtract(subvec_c, element_op(subvec_a)), 2).sum() == 0


@pytest.fixture(
    params=[
        5.0,
        5,
        np.float64(2.0),
        # , 5, 0
    ]
)
def setup_scalar(request):
    return request.param


def test_scalar_mul(setup_scalar, setup_vec):
    vec, *_ = setup_vec
    alpha = setup_scalar

    def scalar_mul(x):
        return alpha * x

    _test_unary_op(scalar_mul, vec)


def test_vec_set(setup_vec):
    vec, *_ = setup_vec
    vec['a'] = 5
    print(vec['a'])
    assert np.sum(np.power(vec.sub['a'] - 5, 2)) == 0
