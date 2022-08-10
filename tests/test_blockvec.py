import numpy as np
import petsc4py.PETSc as PETSc

from blockarray import blockmat as bmat
from blockarray import linalg as bla
from blockarray import blockvec as bvec
from blockarray import blockarray as btensor

# pylint: disable=unused-import
# pylint: disable=missing-function-docstring

a = np.arange(2)
b = np.arange(3)
c = np.arange(4)
VEC1 = bvec.BlockVector((a, b, c), labels=(('a', 'b', 'c'),))

a = np.arange(2)+1
b = np.arange(3)+1
c = np.arange(4)+1
VEC2 = bvec.BlockVector((a, b, c), labels=(('a', 'b', 'c'),))

VEC3 = bvec.BlockVector((a, b, c))

def test_size_shape():
    print(VEC1.size)
    print(VEC1.f_shape)
    print(VEC1.mshape)
    print(VEC1.f_bshape)
    VEC1.print_summary()
    assert VEC1.size == 3
    assert VEC1.f_shape == (3,)
    assert VEC1.mshape == (2+3+4, )
    assert VEC1.f_bshape == ((2, 3, 4),)

def _test_binary_op(op, vec_a, vec_b, element_op=None):
    """
    Tests a binary operation against the equivalent operation on the subtensors
    """
    element_op = op if element_op is None else element_op
    vec_c = op(vec_a, vec_b)
    for subvec_c, subvec_a, subvec_b in zip(vec_c.sub[:], vec_a.sub[:], vec_b.sub[:]):
        assert np.all(subvec_c == element_op(subvec_a, subvec_b))

def test_add():
    _test_binary_op(lambda x, y: x+y, VEC1, VEC2)

def test_div():
    _test_binary_op(lambda x, y: x/y, VEC1, VEC2)

def test_mul():
    _test_binary_op(lambda x, y: x*y, VEC1, VEC2)

def test_power():
    _test_binary_op(btensor.power, VEC1, VEC2, element_op=lambda x, y: x**y)

def test_scalar_mul():
    alpha = 5.0
    ans = alpha*VEC1
    for vec_ans, vec in zip(ans.sub[:], VEC1.sub[:]):
        assert np.all(vec_ans == alpha*vec)

    alpha = np.float64(5.0)
    ans = alpha*VEC1
    for vec_ans, vec in zip(ans.sub[:], VEC1.sub[:]):
        assert np.all(vec_ans == alpha*vec)

def test_vec_set():
    VEC1['a'] = 5
    assert np.all(VEC1.sub['a'] == 5)
    print(VEC1)

if __name__ == '__main__':
    test_size_shape()
    test_add()
    test_div()
    test_mul()
    test_power()
    test_scalar_mul()
    test_vec_set()