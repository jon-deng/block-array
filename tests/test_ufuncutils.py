from numpy import ufunc
from blocktensor import ufuncutils

SIGNATURE = '(i,j),(j,k)->(i, k)'

def test_parse_ufunc_signature():
    sig = SIGNATURE
    input_sigs, output_sig = ufuncutils.parse_ufunc_signature(sig)

    print(sig)
    print(input_sigs)
    print(output_sig)

def test_interpret_ufunc_signature():
    sig_inputs = [('i', 'j'), ('j', 'k')]
    sig_outputs = [('i', 'k')]
    print(ufuncutils.interpret_ufunc_signature(sig_inputs, sig_outputs))

def test_split_shapes_by_signatures():
    shape_inputs = [(2, 3, 2, 4), (2, 3, 4, 2)]
    sig_inputs = [('i', 'j'), ('j', 'k')]

    print(ufuncutils.split_shapes_by_signatures(shape_inputs, sig_inputs))
    # shape_outputs = []

def test_calculate_output_shapes():
    shape_inputs = [(2, 3, 2, 4), (2, 3, 4, 2)]
    sig_inputs = [('i', 'j'), ('j', 'k')]
    sig_outputs = [('i', 'k')]

    ewise_input_shapes, core_input_shapes = ufuncutils.split_shapes_by_signatures(shape_inputs, sig_inputs)

    ewise_output_shapes, core_output_shapes = ufuncutils.calculate_output_shapes(ewise_input_shapes, core_input_shapes, sig_inputs, sig_outputs)
    print(ewise_output_shapes, core_output_shapes)

def test_gen_in_multi_index():
    shape_inputs = [(2, 3, 2, 4), (2, 3, 4, 2)]
    sig_inputs = [('i', 'j'), ('j', 'k')]
    sig_outputs = [('i', 'k')]

    ewise_input_ndims = [2, 2]

    gen_input_midx = ufuncutils.make_gen_in_multi_index(
        ewise_input_ndims, sig_inputs, sig_outputs[0]
    )

    out_midx = (0, 0, 9, 2)
    print(gen_input_midx(out_midx))


if __name__ == '__main__':
    test_parse_ufunc_signature()
    test_interpret_ufunc_signature()
    test_split_shapes_by_signatures()
    test_calculate_output_shapes()
    test_gen_in_multi_index()
    
