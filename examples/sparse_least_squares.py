"""Solve the L1-penalized least squares problem (also known as basis pursuit denoising, or BPDN),

min_x mu||x||_1 + .5||Ax-b||^2,

using the FASTA solver."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots


def sparse_least_squares(A, At, b, mu, x0, **kwargs):
    """Solve the L1-penalized least squares problem.

    :param A: A matrix or function handle.
    :param At: The transpose of A.
    :param b: A measurement vector.
    :param mu: A parameter controlling the regularization.
    :param x0: An initial guess for the solution.
    :param kwargs: Options for the FASTA solver.
    :return: The output of the FASTA solver on the problem.
    """
    f = lambda z: .5 * la.norm((z - b).ravel())**2
    gradf = lambda z: z - b
    g = lambda x: mu * la.norm(x.ravel(), 1)
    proxg = lambda x, t: proximal.shrink(x, t*mu)

    x = fasta(A, At, f, gradf, g, proxg, x0, **kwargs)

    return x.solution, x


def test():
    # Number of measurements
    M = 200

    # Dimension of sparse signal
    N = 1000

    # Signal sparsity
    K = 10

    # Regularization parameter
    mu = 0.02

    # Noise level in b
    sigma = 0.01

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

    print("Constructed sparse least squares problem.")

    # Test the three different algorithms
    plain, adaptive, accelerated = tests.test_modes(lambda **k: sparse_least_squares(A, A.T, b, mu, x0, **k))
    plots.plot_convergence("Sparse Least Squares",
                           (plain[1], adaptive[1], accelerated[1]), ("Plain", "Adaptive", "Accelerated"))

    # Plot the recovered signal
    plots.plot_signals("Sparse Least Squares Regression", x, adaptive[0])
    plots.show_plots()

if __name__ == "__main__":
    test()