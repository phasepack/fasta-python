import numpy as np


def project_Linf_ball(z, t):
    """Project a vector onto a L-inf ball of radius t."""

    N = len(z)
    zabs = np.abs(z)

    # Reverse sort the absolute values of z
    flipped = zabs.copy()
    flipped[::-1].sort()

    # Magic
    alpha = np.max((np.cumsum(flipped) - t) / np.arange(1, N+1))

    if alpha > 0:
        return np.minimum(zabs, alpha) * np.sign(z)
    else:
        return np.zeros(N)


def project_L1_ball(z, t):
    """Project a vector onto a L1 ball of radius t."""

    # By Moreau's identity, we convert to proximal of conjugate problem (L-inf norm)
    return z - project_Linf_ball(z, t)


def shrink(z, t):
    """The vector shrink operator, which is also the proximal operator for the L1-norm."""

    return np.sign(z) * np.maximum(np.abs(z) - t, 0)