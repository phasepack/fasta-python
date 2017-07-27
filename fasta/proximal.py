"""Various common proximal operators of functions."""

import numpy as np
from numpy import linalg as la

__author__ = "Noah Singer"

__all__ = ["project_Linf_ball", "project_L1_ball", "project_Lnuc_ball", "shrink"]


def project_Linf_ball(x: np.ndarray, t: float):
    """Project a vector onto an L-inf ball.

    :param x: The vector to project
    :param t: The radius of the L-inf ball
    """
    N = len(x)
    xabs = np.abs(x)

    # Reverse sort the absolute values of z
    flipped = xabs.copy()
    flipped[::-1].sort()

    # Magic
    alpha = np.max((np.cumsum(flipped) - t) / np.arange(1, N+1))

    if alpha > 0:
        return np.minimum(xabs, alpha) * np.sign(x)
    else:
        return np.zeros(N)


def project_L1_ball(x: np.ndarray, t: float):
    """Project a vector onto an L1 ball.

    :param x: The vector to project
    :param t: The radius of the L1 ball
    """
    # By Moreau's identity, we convert to proximal of dual problem (L-inf norm)
    return x - project_Linf_ball(x, t)


def project_Lnuc_ball(X: np.ndarray, t: float):
    """Project a matrix onto a ball induced by the nuclear norm.

    :param x: The matrix to project
    :param t: The radius of the L-nuc ball
    """
    U, s, V = la.svd(X)

    # Construct the diagonal matrix of singular values, S, as a shrunken version of the original signal values
    S = np.zeros(X.shape)
    S[:len(s),:len(s)] = np.diag(shrink(s, t))
    return U @ S @ V


def shrink(x: np.ndarray, t: float):
    """The shrink (soft-thresholding) operator, which is also the proximal operator for the L1-norm.

    The shrink operator reducing the magnitudes of all entries in x by t, leaving them at zero if they're already less
    than t.

    :param x: The vector to shrink (also could be a matrix)
    :param t: The amount to shrink by
    """
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)
