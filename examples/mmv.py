"""Solve the multiple measurement vector (MMV) problem,

min_X mu*MMV(X) + .5||AX-B||^2,

using the FASTA solver. X is a matrix, and so the norm ||AX-B|| is the Frobenius norm.
The problem assumes that each column has the same sparsity pattern, and so the sparsity constraint on the matrix X
is formulated as,

MMV(X) = sum_i ||X_i||,

where X_i denotes the ith row of X."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots


def mmv(A, At, B, mu, X0, **kwargs):
    """Solve the multiple measurement vector (MMV) problem.

    :param A: A matrix or function handle.
    :param At: The transpose of A.
    :param B: A matrix of measurements.
    :param mu: A parameter controlling the regularization.
    :param X0: An initial guess for the solution.
    :param kwargs: Options for the FASTA solver.
    :return: The output of the FASTA solver on the problem.
    """
    f = lambda Z: .5 * la.norm((Z-B).ravel())**2
    gradf = lambda Z: Z-B
    g = lambda X: mu * np.sum(np.sqrt(np.sum(X*X, axis=1)))

    def proxg(X, t):
        norms = la.norm(X, axis=1)

        # Shrink the norms, and ensure we don't divide by zero
        scale = proximal.shrink(norms, t) / (norms + (norms == 0))

        return X * scale[:,np.newaxis]

    X = fasta(A, At, f, gradf, g, proxg, X0, **kwargs)

    return X.solution, X


def test():
    # Number of measurements
    M = 20

    # Dimension of sparse signal
    N = 30
    L = 10

    # Signal sparsity
    K = 7

    # Regularization parameter
    mu = 1

    # Noise level in b
    sigma = 0.1

    # Create sparse signal
    X = np.zeros((N, L))
    X[np.random.permutation(N)[:K],] = np.random.randn(K, L)

    # Create matrix
    A = np.random.randn(M, N)

    # Create noisy observation matrix
    B = A @ X + sigma * np.random.randn(M, L)

    # Initial iterate
    X0 = np.zeros((N, L))

    print("Constructed MMV problem.")

    # Test the three different algorithms
    plain, adaptive, accelerated = tests.test_modes(lambda **k: mmv(A, A.T, B, mu, X0, **k))
    plots.plot_convergence("Multiple Measurement Vector",
                           (plain[1], adaptive[1], accelerated[1]), ("Plain", "Adaptive", "Accelerated"))

    # Plot the recovered signal
    plots.plot_matrices("Multiple Measurement Vector Recovery", X, adaptive[0])
    plots.show_plots()

if __name__ == "__main__":
    test()