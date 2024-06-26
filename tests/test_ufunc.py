"""
Test `ufunc.py` functionality
"""

import pytest

import numpy as np
from blockarray import blockarray as btensor, ufunc

# pylint: disable=missing-function-docstring


@pytest.fixture(
    params=[
        ('(i)->()', [('i',)], [()]),
        ('(i),(i)->()', [('i',), ('i',)], [()]),
        ('(i,j),(j,k)->(i,k)', [('i', 'j'), ('j', 'k')], [('i', 'k')]),
        ('(j,i),(k,j)->(i,k)', [('j', 'i'), ('k', 'j')], [('i', 'k')]),
    ]
)
def setup_input_output_signatures(request):
    sig_str, in_sigs, out_sigs = request.param
    return sig_str, in_sigs, out_sigs


def test_parse_ufunc_signature(setup_input_output_signatures):
    sig_str, in_sigs, out_sigs = setup_input_output_signatures
    input_sigs, output_sigs = ufunc.parse_ufunc_signature(sig_str)

    assert input_sigs == in_sigs and output_sigs == out_sigs


@pytest.fixture(
    params=[
        ([('i',)], [()], {}, {'i': [(0, 0)]}),
        ([('i',), ('i',)], [()], {}, {'i': [(0, 0), (1, 0)]}),
        (
            [('i', 'j'), ('j', 'k')],
            [('i', 'k')],
            {'i': [(0, 0)], 'k': [(1, 1)]},
            {'j': [(0, 1), (1, 0)]},
        ),
        (
            [('j', 'i'), ('k', 'j')],
            [('i', 'k')],
            {'i': [(0, 1)], 'k': [(1, 0)]},
            {'j': [(0, 0), (1, 1)]},
        ),
    ]
)
def setup_signature_descr(request):
    sig_ins, sig_outs, free_name_descr, redu_name_descr = request.param
    return sig_ins, sig_outs, free_name_descr, redu_name_descr


def test_interpret_ufunc_signature(setup_signature_descr):
    sig_ins, sig_outs, free_name_descr, redu_name_descr = setup_signature_descr
    _free_name_descr, _redu_name_descr = ufunc.interpret_ufunc_signature(
        sig_ins, sig_outs
    )

    assert free_name_descr == _free_name_descr
    assert redu_name_descr == _redu_name_descr


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


class TestBroadcast:
    def test_a(self):
        a = (1, 2, 3, (4, 5), (5, 6))
        b = ((2, 3), 2, 1, (4, 5), (1, 1))
        c = ufunc.broadcast_axis_size(a, b)
        assert c == ((2, 3), 2, 3, (4, 5), (5, 6))

    def test_b(self):
        a = (1, 2, 3, 1, (5, 6))
        b = (10, (2, 2), 1, (4, 5), (1,))
        c = ufunc.broadcast_axis_size(a, b)
        assert c == (10, (2, 2), 3, (4, 5), (5, 6))

    def test_c(self):
        a = (1, (1, (1,)))
        b = (5, (3, 4))
        c = ufunc.broadcast_axis_size(a, b)
        assert c == (5, (3, (4,)))

    def test_d(self):
        a = (1, 1, 2)
        b = (6, 5, 4, 2)
        d = ufunc.broadcast(ufunc.broadcast_axis_size, a, b)
        assert d == (6, 5, 4, 2)

    def test_e(self):
        a = (1, 1, 2)
        b = (6, 5, 4, 2)
        c = (4, 1)
        d = ufunc.broadcast(ufunc.broadcast_axis_size, a, b, c)
        assert d == (6, 5, 4, 2)

    def test_f(self):
        # Expect this case to not work
        with pytest.raises(ValueError) as exc:
            a = ((5, 4), (2, 2), (1, 4))
            b = (6, (5, 1), (3, 2, 2), (4, 1))
            c = (4,)
            d = ufunc.broadcast(ufunc.broadcast_size, a, b, c)
            print(d)


class TestUfunc:
    @pytest.fixture(
        params=[
            np.add,
            np.subtract,
            np.multiply,
            np.divide,
            np.power,
            # np.matmul
            # This one should be it's own special thing since it's not a
            # simple signature
        ]
    )
    def setup_ufunc(self, request):
        """
        Return a pre-defined `LabelledArray` and reference data
        """
        return request.param

    @pytest.fixture(
        params=[
            # [(5,), ((5,))],
            # [(5,), ((5,))],
            [((2, 4), (2, 4)), ((2, 4), (2, 4))],
            [((2, 4), (1,), (1,)), ((1,), (1,))],
            [((2, 4), (3,), (1,)), ((1,), (3,))],
            # ( ((2, 4), (2, 4), (1, 1)), ((1, 4), (1, 1)) )
        ]
    )
    def setup_binary_inputs(self, request):
        A = btensor.rand(request.param[0])
        B = btensor.rand(request.param[1])
        return A, B

    def test_apply_binary_ufunc(self, setup_ufunc, setup_binary_inputs):
        """Test binary ufuncs"""
        A, B = setup_binary_inputs
        _ufunc = setup_ufunc

        D = ufunc.apply_ufunc_array(_ufunc, '__call__', *[A, B])
        D_ = _ufunc(A.to_mono_ndarray(), B.to_mono_ndarray())

        assert compare_blockarray_to_monoarray(D, D_)

    @pytest.fixture(
        params=[((2,),), ((5, 5, 5),), ((2, 4), (2, 4)), (2, (2, 4)), ((1,), (2, 3), 4)]
    )
    def setup_reduce_inputs(self, request):
        A = btensor.rand(request.param)
        return A

    def test_apply_ufunc_reduce(self, setup_reduce_inputs):
        A = setup_reduce_inputs

        # Reducing the 2d array gives a 1d array
        D = np.add.reduce(A, axis=-1)
        D_ = np.add.reduce(A.to_mono_ndarray(), axis=-1)

        assert compare_blockarray_to_monoarray(D, D_)

    @pytest.fixture(
        params=[((2,),), ((5, 5, 5),), ((2, 4), (2, 4)), (2, (2, 4)), ((1,), (2, 3), 4)]
    )
    def setup_accumulate_inputs(self, request):
        A = btensor.rand(request.param)
        return A

    def test_apply_ufunc_accumulate(self, setup_accumulate_inputs):
        A = setup_accumulate_inputs

        D = np.add.accumulate(A, axis=-1)
        D_ = np.add.accumulate(A.to_mono_ndarray(), axis=-1)
        print(D, D_)
        assert compare_blockarray_to_monoarray(D, D_)


def compare_blockarray_to_monoarray(barray, marray):
    if isinstance(barray, btensor.BlockArray):
        return np.all(np.isclose(barray.to_mono_ndarray(), marray))
    else:
        return np.all(np.isclose(barray, marray))
