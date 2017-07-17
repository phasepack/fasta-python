"""Solve the L1-restricted least squares problem (also known as the LASSO problem),

min_x .5||Ax-b||^2, ||x||_1 < mu,

using the FASTA solver. The problem is re-expressed with a characteristic function function for the constraint."""

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots

__author__ = "Noah Singer"


def lasso(A, At, b, mu, x0, **kwargs):
    """Solve the L1-restricted least squares problem.

    :param A: A matrix or function handle
    :param At: The transpose of A
    :param b: A measurement vector
    :param mu: A parameter controlling the regularization
    :param x0: An initial guess for the solution
    :param kwargs: Options for the FASTA solver
    :return: The problem's computed solution and the full output of the FASTA solver on the problem
    """
    f = lambda z: .5 * la.norm((z - b).ravel())**2
    gradf = lambda z: z - b
    g = lambda x: 0 # TODO: add an extra condition to this
    proxg = lambda x, t: proximal.project_L1_ball(x, mu)

    x = fasta(A, At, f, gradf, g, proxg, x0, **kwargs)

    return x.solution, x


def test(M=200, N=1000, K=10, sigma=0.01, mu=0.8):
    """Construct a sample LASSO regression problem with a random sparse signal and measurement matrix.

    :param M: The number of measurements (default: 200)
    :param N: The dimension of the sparse signal (default: 1000)
    :param K: The signal sparsity (default: 10)
    :param sigma: The noise level in the observation vector (default: 0.01)
    :param mu: The regularization parameter (default: 0.8)
    """
    # Create sparse signal
    x = np.zeros(N)
    x[np.random.permutation(N)[:K]] = 1

    # Normalize the regularization parameter
    mu *= la.norm(x, 1)

    # Create matrix
    A = np.random.randn(M, N)
    A /= la.norm(A, 2)

    # Create noisy observation vector
    b = A @ x + sigma * np.random.randn(M)

    # Initial iterate
    x0 = np.zeros(N)

    print("Constructed LASSO problem.")

    # Test the three different algorithms
    adaptive, accelerated, plain = tests.test_modes(lambda **k: lasso(A, A.T, b, mu, x0, **k))
    plots.plot_convergence("LASSO",
                           (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))

    # Plot the recovered signal
    plots.plot_signals("LASSO Regression", x, adaptive[0])
    plots.show_plots()

if __name__ == "__main__":
    test()

del np, la
del fasta, tests, proximal, plots
