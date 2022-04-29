
import numpy as np
from blockarray import blockarray as btensor, ufunc

SIGNATURE = '(i,j),(j,k)->(i, k)'

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
        ewise_input_ndims, sig_inputs, sig_outputs[0]
    )

    print(gen_input_midx(out_midx))

def test_recursive_concatenate():
    a = np.ones((4, 4))
    b = np.ones((4, 2))
    c = np.ones((2, 4))
    d = np.ones((2, 2))
    A = btensor.BlockArray([[a, b], [c, d]])

    ufunc.recursive_concatenate(A.subarrays_flat, A.shape, A.dims)

def test_apply_ufunc():
    a = np.ones((4, 4))
    b = np.ones((4, 2))
    c = np.ones((2, 4))
    d = np.ones((2, 2))
    A = btensor.BlockArray([[a, b], [c, d]])

    a = np.ones((4, 4))
    b = np.ones((4, 2))
    c = np.ones((2, 4))
    d = np.ones((2, 2))
    B = btensor.BlockArray([[a, b], [c, d]])

    # C = ufuncutils.apply_ufunc(np.add, '__call__', *[A, B])
    # print(C.shape)

    D = ufunc.apply_ufunc(np.matmul, '__call__', *[A, B])
    D_ = np.matmul(A.to_mono_ndarray(), B.to_mono_ndarray())
    assert np.all(np.isclose(D.to_mono_ndarray(), D_))

    D = ufunc.apply_ufunc(np.add, '__call__', *[A, B])
    D_ = np.add(A.to_mono_ndarray(), B.to_mono_ndarray())
    assert np.all(np.isclose(D.to_mono_ndarray(), D_))

    scalar_np = np.float64(5.0)
    D = ufunc.apply_ufunc(np.add, '__call__', *[scalar_np, B])
    D_ = np.add(scalar_np, B.to_mono_ndarray())
    assert np.all(np.isclose(D.to_mono_ndarray(), D_))

    D = ufunc.apply_ufunc(np.multiply, '__call__', *[scalar_np, B])
    D_ = np.multiply(scalar_np, B.to_mono_ndarray())
    assert np.all(np.isclose(D.to_mono_ndarray(), D_))

if __name__ == '__main__':
    test_parse_ufunc_signature()
    test_interpret_ufunc_signature()
    test_split_shapes_by_signatures()
    test_calculate_output_shapes()
    test_gen_in_multi_index()
    test_recursive_concatenate()
    test_apply_ufunc()
