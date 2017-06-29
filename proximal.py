import numpy as np


def project_Linf_ball(x, t):
    """Project a vector onto a L-inf ball of radius t."""

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


def project_L1_ball(x, t):
    """Project a vector onto a L1 ball of radius t."""

    # By Moreau's identity, we convert to proximal of conjugate problem (L-inf norm)
    return x - project_Linf_ball(x, t)


def shrink(x, t):
    """The vector shrink operator, which is also the proximal operator for the L1-norm."""

    return np.sign(x) * np.maximum(np.abs(x) - t, 0)