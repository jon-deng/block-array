
import numpy as np
from blockarray import blockarray as btensor, ufunc

SIGNATURE = '(i,j),(j,k)->(i, k)'

# pylint: disable=missing-function-docstring

def test_parse_ufunc_signature():
    sig = SIGNATURE
    input_sigs, output_sig = ufunc.parse_ufunc_signature(sig)

    print(sig)
    print(input_sigs)
    print(output_sig)

def test_interpret_ufunc_signature():
    sig_inputs = [('i', 'j'), ('j', 'k')]
    sig_outputs = [('i', 'k')]
    print(ufunc.interpret_ufunc_signature(sig_inputs, sig_outputs))

def test_split_shapes_by_signatures():
    shape_inputs = [(2, 3, 2, 4), (2, 3, 4, 2)]
    sig_inputs = [('i', 'j'), ('j', 'k')]

    print(ufunc.split_shapes_by_signatures(shape_inputs, sig_inputs))
    # shape_outputs = []

def test_calculate_output_shapes():
    shape_inputs = [(2, 3, 2, 4), (2, 3, 4, 2)]
    sig_inputs = [('i', 'j'), ('j', 'k')]
    sig_outputs = [('i', 'k')]

    ewise_input_shapes, core_input_shapes = ufunc.split_shapes_by_signatures(shape_inputs, sig_inputs)

    ewise_output_shapes, core_output_shapes = ufunc.calculate_output_shapes(ewise_input_shapes, core_input_shapes, sig_inputs, sig_outputs)
    print(ewise_output_shapes, core_output_shapes)

def test_gen_in_multi_index():
    sig_inputs = [('i', 'j'), ('j', 'k')]
    sig_outputs = [('i', 'k')]

    shape_inputs = [(2, 3, 2, 4), (2, 3, 4, 2)]
    ewise_input_ndims = [2, 2]
    out_midx = (0, 0, 9, 2)

    shape_inputs = [(2, 4), (4, 2)]
    ewise_input_ndims = [0, 0]
    out_midx = (9, 2)

    gen_input_midx = ufunc.make_gen_in_multi_index(
        shape_inputs, sig_inputs, sig_outputs[0]
    )

    print(gen_input_midx(out_midx))

def test_broadcast():
    a = (     1, 2, 3, (4, 5), (5, 6))
    b = ((2, 3), 2, 1, (4, 5), (1, 1))
    c = ufunc.broadcast_axis_size(a, b)
    assert c == ((2, 3), 2, 3, (4, 5), (5, 6))

    a = ( 1,     2,  3,      1, (5, 6))
    b = (10, (2, 2), 1, (4, 5),   (1,))
    c = ufunc.broadcast_axis_size(a, b)
    assert c == (10, (2, 2), 3, (4, 5), (5, 6))

    a = (1, (1, (1,)))
    b = (5, (3,    4))
    c = ufunc.broadcast_axis_size(a, b)
    assert c == (5, (3, (4,)))

    a =    (1, 1, 2)
    b = (6, 5, 4, 2)
    c =       (4, 1)
    d = ufunc.broadcast(ufunc.broadcast_axis_size, (a, b, c))
    print(d)

    a =    ((5, 4), (2, 2), (1, 4))
    b = (6, (5, 1), (2, 2), (4, 1))
    c =                        (4,)
    d = ufunc.broadcast(ufunc.broadcast_axis_size, (a, b, c))
    print(d)

    # Expect this case to not work
    # a =    ((5, 4),    (2, 2), (1, 4))
    # b = (6, (5, 1), (3, 2, 2), (4, 1))
    # c =                        (4,)
    # d = ufunc.broadcast(ufunc.broadcast_size, a, b, c)
    # print(d)

# def test_recursive_concatenate():
#     a = np.ones((4, 4))
#     b = np.ones((4, 2))
#     c = np.ones((2, 4))
#     d = np.ones((2, 2))
#     A = btensor.BlockArray([[a, b], [c, d]])

#     ufunc.recursive_concatenate(A.subarrays_flat, A.shape, A.dims)

def test_apply_ufunc():
    ## Test binary ufuncs with 2D inputs
    a = np.random.random_sample((4, 4))
    b = np.random.random_sample((4, 2))
    c = np.random.random_sample((2, 4))
    d = np.random.random_sample((2, 2))
    A = btensor.BlockArray([[a, b], [c, d]])

    a = np.random.random_sample((4, 4))
    b = np.random.random_sample((4, 2))
    c = np.random.random_sample((2, 4))
    d = np.random.random_sample((2, 2))
    B = btensor.BlockArray([[a, b], [c, d]])

    # Test the classic example matmul(A, B)
    D = ufunc.apply_ufunc_array(np.matmul, '__call__', *[A, B])
    D_ = np.matmul(A.to_mono_ndarray(), B.to_mono_ndarray())
    assert np.all(np.isclose(D.to_mono_ndarray(), D_))

    # Test an example with collapsed axes matmul(A[0, :], B[:, 0])
    D = ufunc.apply_ufunc_array(np.matmul, '__call__', *[A[0, :], B[:, 0]])
    D_ = np.matmul(A[0, :].to_mono_ndarray(), B[:, 0].to_mono_ndarray())
    assert np.all(np.isclose(D.to_mono_ndarray(), D_))

    D = ufunc.apply_ufunc_array(np.add, '__call__', *[A, B])
    D_ = np.add(A.to_mono_ndarray(), B.to_mono_ndarray())
    assert np.all(np.isclose(D.to_mono_ndarray(), D_))

    scalar_np = np.float64(5.0)
    D = ufunc.apply_ufunc_array(np.add, '__call__', *[scalar_np, B])
    D_ = np.add(scalar_np, B.to_mono_ndarray())
    assert np.all(np.isclose(D.to_mono_ndarray(), D_))

    D = ufunc.apply_ufunc_array(np.multiply, '__call__', *[scalar_np, B])
    D_ = np.multiply(scalar_np, B.to_mono_ndarray())
    assert np.all(np.isclose(D.to_mono_ndarray(), D_))

    a = np.random.random_sample((4, 4, 1, 1))
    b = np.random.random_sample((4, 2, 1, 1))
    c = np.random.random_sample((2, 4, 1, 1))
    d = np.random.random_sample((2, 2, 1, 1))
    A = btensor.BlockArray([a, b, c, d], shape=(2, 2, 1, 1))

    a = np.random.random_sample((4, 4, 1, 1))
    b = np.random.random_sample((4, 2, 1, 1))
    c = np.random.random_sample((2, 4, 1, 1))
    d = np.random.random_sample((2, 2, 1, 1))
    B = btensor.BlockArray([a, b, c, d], shape=(2, 2, 1, 1))

    D = ufunc.apply_ufunc_array(
        np.matmul, '__call__', *[A, B], axes=[(0, 1), (0, 1), (-2, -1)]
    )
    D_ = np.matmul(A.to_mono_ndarray(), B.to_mono_ndarray(), axes=[(0, 1), (0, 1), (-2, -1)])
    assert np.all(np.isclose(D.to_mono_ndarray(), D_))

    D_ = np.matmul(A.to_mono_ndarray(), B.to_mono_ndarray(), axes=[(0, 1), (0, 1), (-2, -1)])
    assert np.all(np.isclose(D.to_mono_ndarray(), D_))

    # Unfortunately, seems like inner1d is not available in the numpy public api?
    # D = ufunc.apply_ufunc_array(
    #     np.inner1d, '__call__', *[A, B], axes=[(1,), (0,), ()])
    # D_ = np.inner1d(A.to_mono_ndarray(), B.to_mono_ndarray(), axes=[(1,), (0,), ()])
    # assert np.all(np.isclose(D.to_mono_ndarray(), D_))

def test_apply_ufunc_reduce():
    a = np.random.random_sample((4, 4))
    b = np.random.random_sample((4, 2))
    c = np.random.random_sample((2, 4))
    d = np.random.random_sample((2, 2))
    A = btensor.BlockArray([[a, b], [c, d]])

    a = np.random.random_sample((4, 4))
    b = np.random.random_sample((4, 2))
    c = np.random.random_sample((2, 4))
    d = np.random.random_sample((2, 2))
    B = btensor.BlockArray([[a, b], [c, d]])

    # Reducing the 2d array gives a 1d array
    D = np.add.reduce(A)
    D_ = np.add.reduce(A.to_mono_ndarray())
    np.all(np.isclose(D.to_mono_ndarray(), D_))

    # Reducing the 1d array should give a 0d array (scalar)
    E = np.add.reduce(D)
    E_ = np.add.reduce(D.to_mono_ndarray())
    np.all(np.isclose(E.to_mono_ndarray(), E_))

def test_apply_ufunc_accumulate():
    a = np.random.random_sample((4, 4))
    b = np.random.random_sample((4, 2))
    c = np.random.random_sample((2, 4))
    d = np.random.random_sample((2, 2))
    A = btensor.BlockArray([[a, b], [c, d]])

    a = np.random.random_sample((4, 4))
    b = np.random.random_sample((4, 2))
    c = np.random.random_sample((2, 4))
    d = np.random.random_sample((2, 2))
    B = btensor.BlockArray([[a, b], [c, d]])

    D = np.add.accumulate(A)
    D_ = np.add.accumulate(A.to_mono_ndarray())
    np.all(np.isclose(D.to_mono_ndarray(), D_))


if __name__ == '__main__':
    test_parse_ufunc_signature()
    test_interpret_ufunc_signature()
    test_split_shapes_by_signatures()
    test_calculate_output_shapes()
    test_gen_in_multi_index()
    # test_recursive_concatenate()
    test_apply_ufunc()
    test_apply_ufunc_reduce()
    test_apply_ufunc_accumulate()

    test_broadcast()
