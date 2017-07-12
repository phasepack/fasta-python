"""A collection of various utility functions for PhasePack and FASTA."""

__author__ = "Noah Singer"

import numpy as np


def functionize(A):
    """Check if an object A is a function. If it's not, return a function wrapping A.

    :param A: an object (possibly already a function).
    :return: a function returning A, if it's not a function.
    """

    if A is None:
        # A is simply the identity
        return lambda x: x
    elif not callable(A):
        # A is not a function, so create a function wrapping it
        return lambda x: A @ x
    else:
        # A is already a function, so just return it
        return A

#
# def check_adjoint(A, At, x):
#     x = np.random.randn(len(x))
#     Ax = A(x)
#
#     y = np.random.randn(len(Ax))
#     Aty = At(y)
