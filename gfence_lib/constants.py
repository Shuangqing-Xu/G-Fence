BASE = 2
PRECISION_INTEGRAL = 0
PRECISION_FRACTIONAL = 10
Q = 2**61 - 1
PRECISION = PRECISION_INTEGRAL + PRECISION_FRACTIONAL
INVERSE = pow(BASE**PRECISION_FRACTIONAL, Q - 2, Q)

assert Q > BASE**PRECISION

def inverse_for_precision(precision_fractional):
    return pow(BASE**precision_fractional, Q - 2, Q)

