"""Solve the L1-penalized least squares problem (also known as basis pursuit denoising, or BPDN),

min mu||x||_1 + .5||Ax-b||^2

using the FASTA solver."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots


def sparse_logistic(A, At, b, mu, x0, **kwargs):
    """Solve the L1-penalized logistic least squares problem.

    :param A: A matrix or function handle.
    :param At: The transpose of A.
    :param b: A measurement vector.
    :param mu: A parameter controlling the regularization.
    :param x0: An initial guess for the solution.
    :param kwargs: Options for the FASTA solver.
    :return: The output of the FASTA solver on the problem.
    """
    f = lambda z: np.sum(np.log(1 + np.exp(z)) - (b==1) * z)
    gradf = lambda z: -b / (1 + np.exp(b * z))
    g = lambda x: mu * la.norm(x, 1)
    proxg = lambda x, t: proximal.shrink(x, t*mu)

    return fasta(A, At, f, gradf, g, proxg, x0, **kwargs)

if __name__ == "__main__":
    # Number of measurements
    M = 1000

    # Dimension of sparse signal
    N = 2000

    # Signal sparsity
    K = 5

    # Regularization parameter
    mu = 40

    # Create sparse signal
    x = np.zeros(N)
    x[np.random.permutation(N)[:K]] = 1

    # Create matrix
    A = np.random.randn(M, N)

    # Create observation vector
    probabilities = 1 / (1 + np.exp(-A @ x))
    b = 2.0 * (np.random.rand(M) < probabilities) - 1

    # Initial iterate
    x0 = np.zeros(N)

    print("Constructed sparse logistic least-squares problem.")

    # Test the three different algorithms
    plain, adaptive, accelerated = tests.test_modes(lambda **k: sparse_logistic(A, A.T, b, mu, x0, **k))

    # Plot the recovered signal
    plots.plot_signals(x, adaptive.solution)
    plots.show_plots()