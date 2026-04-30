# Constitutive utilities: strain, stress, and complex inner products.

from ngsolve import *
import numpy as np


def strain(w):
    return Sym(Grad(w))

def stress(w, lam, mu, dim=2):
    aa = strain(w)
    return 2 * mu * aa + lam * Trace(aa) * Id(dim)


def complexify(trial_func: tuple, test_func: tuple):
    """Re(a*conj(b)) split into real/imag blocks for saddle systems."""
    real_part = InnerProduct(trial_func[0], test_func[0]) + InnerProduct(trial_func[1], test_func[1])
    imag_part = InnerProduct(trial_func[0], test_func[1]) - InnerProduct(trial_func[1], test_func[0])
    return real_part + imag_part

def complexify_multi(trial_func: tuple, test_func: tuple):
    """Scalar version of complexify (no InnerProduct)."""
    real_part = trial_func[0]*test_func[0] + trial_func[1]*test_func[1]
    imag_part = trial_func[0]*test_func[1] - trial_func[1]*test_func[0]
    return real_part + imag_part