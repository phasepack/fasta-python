"""A collection of various utility functions for FASTA."""

import numpy as np
from typing import Union

import fasta

__author__ = "Noah Singer"


def operatorize(A: Union["fasta.LinearOperator", np.ndarray, None]) -> "fasta.LinearOperator":
    """Make an object A into an operator (represented as a function by Python).

    :param A: A linear operator, which may already be a function
    :return: A function returning A, if it's not a function
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
