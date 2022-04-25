from blocktensor import ufuncutils

def test_parse_signature():
    sig = '(i,j),(j,k)->(i, k)'
    input_sigs, output_sig = ufuncutils.parse_ufunc_signature(sig)

    print(sig)
    print(input_sigs)
    print(output_sig)


if __name__ == '__main__':
    test_parse_signature()