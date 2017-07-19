"""Solve the non-negative least squares problem, min_x .5||Ax-b||^2, x >= 0, using the FASTA solver."""

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots

__author__ = "Noah Singer"

__all__ = ["non_negative_least_squares", "test"]


def non_negative_least_squares(A, At, b, x0, **kwargs):
    """Solve the non-negative least squares problem.

    :param A: A matrix or function handle
    :param At: The transpose of A
    :param b: A measurement vector
    :param x0: An initial guess for the solution
    :param kwargs: Options for the FASTA solver
    :return: The problem's computed solution and the full output of the FASTA solver on the problem.
    """
    f = lambda z: .5 * la.norm((z - b).ravel())**2
    gradf = lambda z: z - b
    g = lambda x: 0
    proxg = lambda x, t: np.maximum(x, 0)

    x = fasta(A, At, f, gradf, g, proxg, x0, **kwargs)

    return x.solution, x


def test(M=200, N=1000, K=10, sigma=0.005):
    """Construct a sample non-negative least squares problem with a random sparse signal and measurement matrix.

    :param M: The number of measurements (default: 200)
    :param N: The dimension of the sparse signal (default: 1000)
    :param K: The signal sparsity (default: 10)
    :param sigma: The noise level in the observation vector (default: 0.005)
    """
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

    print("Constructed non-negative least squares problem.")

    # Test the three different algorithms
    adaptive, accelerated, plain = tests.test_modes(lambda **k: non_negative_least_squares(A, A.T, b, x0, **k))
    plots.plot_convergence("Non-Negative Least Squares",
                           (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))

    # Plot the recovered signal
    plots.plot_signals("Non-Negative Least Squares Regression", x, adaptive[0])

    return adaptive, accelerated, plain

if __name__ == "__main__":
    test()
    plots.show_plots()
