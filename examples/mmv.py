"""Solve the multiple measurement vector (MMV) problem,

min mu*MMV(X) + .5||AX-B||^2

using the FASTA solver. X is a matrix, and so the norm ||AX-B|| is the Frobenius norm.
The problem assumes that each column has the same sparsity pattern, and so the sparsity constraint on the matrix X
is formulated as,

MMV(x) = sum_i ||X_i||,

where X_i denotes the ith row of X."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots


def mmv(A, B, mu, X0, **kwargs):
    """Solve the multiple measurement vector (MMV) problem.

    :param A: A matrix or function handle.
    :param B: A matrix of measurements.
    :param mu: A parameter controlling the regularization.
    :param X0: An initial guess for the solution.
    :param kwargs: Options for the FASTA solver.
    :return: The output of the FASTA solver on the problem.
    """
    f = lambda Z: .5 * la.norm(Z-B)**2
    gradf = lambda Z: Z-B
    g = lambda X: mu * np.sum(np.sqrt(np.sum(X*X, axis=1)))

    def proxg(X, t):
        norms = np.sqrt(np.sum(X*X, axis=1))
        scale = proximal.shrink(norms, t) / (norms + (norms == 0))

        # Reshape scale to a column vector
        scale = np.reshape(scale, (len(scale), 1))

        scale = np.kron(scale, np.ones((1, X.shape[1])))

        return X * scale

    return fasta(A, A.T, f, gradf, g, proxg, X0, **kwargs)

if __name__ == "__main__":
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
    X = np.zeros((N,L))
    X[np.random.permutation(N)[:K],] = np.random.randn(K,L)

    # Create matrix
    A = np.random.randn(M, N)

    # Create noisy observation matrix
    B = A @ X + sigma * np.random.randn(M,L)

    # Initial iterate
    X0 = np.zeros((N,L))

    print("Constructed MMV problem.")

    # Test the three different algorithms
    raw, adaptive, accelerated = tests.test_modes(lambda **k: mmv(A, B, mu, X0, **k))

    # Plot the recovered signal
    plots.plot_matrices(X, adaptive.solution)
    plots.show_plots()