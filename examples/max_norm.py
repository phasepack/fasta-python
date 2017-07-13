"""Solve the max-norm problem,

min_X <W, X X^T>, ||X||_{2,inf}^2 <= 1

using the FASTA solver. The constraint operates on the max-norm
"""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots


def max_norm(S, mu, X0, **kwargs):
    """Solve the max-norm problem.

    :param S: A square matrix.
    :param mu: A parameter controlling the regularization.
    :param X0: An initial guess for the gradient of the solution.
    :return: The output of the FASTA solver on the problem.
    """

    f = lambda X: np.sum(S * (X @ X.T))
    gradf = lambda X: (S + S.T) @ X
    g = lambda X: 0

    def proxg(X, t):
        norms = la.norm(X, axis=1)

        # Shrink the norms that are too big, and ensure we don't divide by zero
        scale = np.maximum(norms, t) + norms == 0

        return X / scale[:,np.newaxis]

    X = fasta(None, None, f, gradf, g, proxg, X0, **kwargs)

    return X, X.solution


def test():
    # Regularization parameter
    mu = 0.1

    # Noise level in M
    sigma = 0.05

    # Generate an image
    N = 512
    M = scipy.misc.ascent().astype(float)

    # Normalize M
    M /= np.max(M)

    # Add noise
    M += sigma * np.random.randn(N, N)

    # Initial iterate
    Y0 = np.zeros((N, N, 2))

    print("Constructed max-norm problem.")

    # Test the three different algorithms
    plain, adaptive, accelerated = tests.test_modes(lambda **k: max_norm(M, mu, Y0, **k))

    # Plot the recovered signal
    plots.plot_matrices("Total Variation Denoising", M, adaptive[0])
    plots.show_plots()

if __name__ == "__main__":
    test()
