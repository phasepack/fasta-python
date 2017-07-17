"""Solve the L1-penalized logistic least squares problem,

min mu||x||_1 + logit(Ax,b),

using the FASTA solver, where the logistic log-odds function is defined as,

logic(z,b) = sum_i log(1 + e^(z_i)) - b_i * z_i,

where z_i and b_i are the ith rows of z and b, respectively."""

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots

__author__ = "Noah Singer"


def sparse_logistic(A, At, b, mu, x0, **kwargs):
    """Solve the L1-penalized logistic least squares problem.

    :param A: A matrix or function handle
    :param At: The transpose of A
    :param b: A measurement vector
    :param mu: A parameter controlling the regularization
    :param x0: An initial guess for the solution
    :param kwargs: Options for the FASTA solver
    :return: The problem's computed solution and the full output of the FASTA solver on the problem.
    """
    f = lambda z: np.sum(np.log(1 + np.exp(z)) - (b==1) * z)
    gradf = lambda z: -b / (1 + np.exp(b * z))
    g = lambda x: mu * la.norm(x.ravel(), 1)
    proxg = lambda x, t: proximal.shrink(x, t*mu)

    x = fasta(A, At, f, gradf, g, proxg, x0, **kwargs)

    return x.solution, x


def test(M=1000, N=2000, K=5, mu=40):
    """Construct a sample sparse logistic least squares problem with a random sparse signal and measurement matrix.

    :param M: The number of measurements (default: 1000)
    :param N: The dimension of the sparse signal (default: 2000)
    :param K: The signal sparsity (default: 5)
    :param mu: The regularization parameter (default: 40.0)
    """
    # Create sparse signal
    x = np.zeros(N)
    x[np.random.permutation(N)[:K]] = 1

    # Create matrix
    A = np.random.randn(M, N)

    # Create observation vector
    p = 1 / (1 + np.exp(-A @ x))
    b = 2.0 * (np.random.rand(M) < p) - 1

    # Initial iterate
    x0 = np.zeros(N)

    print("Constructed sparse logistic least-squares problem.")

    # Test the three different algorithms
    adaptive, accelerated, plain = tests.test_modes(lambda **k: sparse_logistic(A, A.T, b, mu, x0, **k))
    plots.plot_convergence("Sparse Logistic Least Squares",
                           (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))

    # Plot the recovered signal
    plots.plot_signals("Sparse Logistic Regression", x, adaptive[0])
    plots.show_plots()

if __name__ == "__main__":
    test()

del np, la
del fasta, tests, proximal, plots
