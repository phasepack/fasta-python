"""Solve the L1-restricted least squares problem,

min .5||Ax-b||^2, ||x||_1 < mu

using the FASTA solver. We express this as min f(Ax) + g(x), where f(Ax) = .5||Ax-b||^2
and g(x) = { 0           |x| < mu
           { infinity    otherwise."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots


def lasso(A, At, b, mu, x0, **kwargs):
    """Solve the L1-restricted least squares problem.

    :param A: A matrix or function handle.
    :param At: The transpose of A.
    :param b: A measurement vector.
    :param mu: A parameter controlling the regularization.
    :param x0: An initial guess for the solution.
    :param kwargs: Options for the FASTA solver.
    :return: The output of the FASTA solver on the problem.
    """
    f = lambda z: .5 * la.norm(z - b)**2
    gradf = lambda z: z - b
    g = lambda x: 0
    proxg = lambda x, t: proximal.project_L1_ball(x, mu)

    return fasta(A, At, f, gradf, g, proxg, x0, **kwargs)

if __name__ == "__main__":
    # Number of measurements
    M = 200

    # Dimension of sparse signal
    N = 1000

    # Signal sparsity
    K = 10

    # Noise level in b
    sigma = 0.01

    # Create sparse signal
    x = np.zeros(N)
    x[np.random.permutation(N)[:K]] = 1

    # Regularization parameter
    mu = 0.8 * la.norm(x, 1)

    # Create matrix
    A = np.random.randn(M, N)
    A /= la.norm(A, 2)

    # Create noisy observation vector
    b = A @ x + sigma * np.random.randn(M)

    # Initial iterate
    x0 = np.zeros(N)

    print("Constructed lasso problem.")

    # Test the three different algorithms
    plain, adaptive, accelerated = tests.test_modes(lambda **k: lasso(A, A.T, b, mu, x0, **k))

    # Plot the recovered signal
    plots.plot_signals(x, adaptive.solution)
    plots.show_plots()
