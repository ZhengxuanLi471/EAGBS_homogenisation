# =============================================================================
# Constitutive Utilities for Complex Elasticity
# -----------------------------------------------------------------------------
# - Wraps strain/stress computations reused across solvers, convergence tests,
#   and plotting scripts.
# - Supplies complex inner-product helpers that keep real/imag blocks explicit
#   for saddle systems and post-processing.
#
# Author: Zhengxuan Li
# Updated: 2 Dec 2025
# =============================================================================


from ngsolve import *
import numpy as np

# Most part of earth interior has Poisson ratio around 0.3


def strain(w):
    return Sym(Grad(w))
def stress(w, lam, mu, dim=2):
    aa = strain(w)
    return 2 * mu * aa + lam * Trace(aa) * Id(dim)


def complexify(trial_func:tuple[any, any] , test_func:tuple[any, any]):
    #returns a multiplication of two complex functions in terms of their real and imaginary parts
    real_part = InnerProduct(trial_func[0], test_func[0]) + InnerProduct(trial_func[1], test_func[1])
    imag_part = InnerProduct(trial_func[0], test_func[1]) - InnerProduct(trial_func[1], test_func[0])
    return real_part+imag_part

def complexify_multi(trial_func:tuple[any, any] , test_func:tuple[any, any]):
    #returns a multiplication of two complex functions in terms of their real and imaginary parts
    real_part = trial_func[0]*test_func[0] + trial_func[1]*test_func[1]
    imag_part = trial_func[0]*test_func[1] - trial_func[1]*test_func[0]
    return real_part+imag_part