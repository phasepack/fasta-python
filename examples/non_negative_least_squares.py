"""Solve the non-negative least squares problem,

min_x .5||Ax-b||^2, x >= 0

using the FASTA solver."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots


def non_negative_least_squares(A, At, b, x0, **kwargs):
    """Solve the non-negative least squares problem.

    :param A: A matrix or function handle.
    :param At: The transpose of A.
    :param b: A measurement vector.
    :param x0: An initial guess for the solution.
    :param kwargs: Options for the FASTA solver.
    :return: The output of the FASTA solver on the problem.
    """
    f = lambda z: .5 * la.norm((z - b).ravel())**2
    gradf = lambda z: z - b
    g = lambda x: 0 if (x >= -1E-12).all() else np.inf
    proxg = lambda x, t: np.maximum(x, 0)

    x = fasta(A, At, f, gradf, g, proxg, x0, **kwargs)

    return x.solution, x

if __name__ == "__main__":
    # Number of measurements
    M = 200

    # Dimension of sparse signal
    N = 1000

    # Signal sparsity
    K = 10

    # Noise level in b
    sigma = 0.005

    # Create sparse signal
    x = np.zeros(N)
    x[np.random.permutation(N)[:K]] = 1

    # Create matrix
    A = np.random.randn(M, N)
    A /= la.norm(A, 2)

    # Create noisy observation vector
    b = A @ x + sigma * np.random.randn(M)

    # Initial iterate
    x0 = np.zeros(N)

    print("Constructed sparse least-squares problem.")

    # Test the three different algorithms
    plain, adaptive, accelerated = tests.test_modes(lambda **k: non_negative_least_squares(A, A.T, b, x0, **k))

    # Plot the recovered signal
    plots.plot_signals("Non-Negative Least Squares Regression", x, adaptive[0])
    plots.show_plots()