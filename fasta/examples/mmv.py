"""Solve the multiple measurement vector (MMV) problem, min_X mu*MMV(X) + .5||AX-B||^2, using the FASTA solver.

X is a matrix, and so the norm ||AX-B|| is the Frobenius norm. The problem assumes that each column of X has the same
sparsity pattern, and so the sparsity constraint on X is formulated as,

    MMV(X) = sum_i ||X_i||,

where X_i denotes the ith row of X."""

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots

__author__ = "Noah Singer"

__all__ = ["mmv", "test"]


def mmv(A, At, B, mu, X0, **kwargs):
    """Solve the multiple measurement vector (MMV) problem.

    :param A: A matrix or function handle
    :param At: The transpose of A
    :param B: A matrix of measurements
    :param mu: A parameter controlling the regularization
    :param X0: An initial guess for the solution
    :param kwargs: Options for the FASTA solver
    :return: The problem's computed solution and the full output of the FASTA solver on the problem
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


def test(M=20, N=30, L=10, K=7, sigma=0.1, mu=1.0):
    """Construct a sample max-norm problem by creating a two-moons segmentation dataset, converting it to a weighted graph, and then performing max-norm regularization on its adjacency matrix.

    :param M: The number of measurements (default: 20)
    :param N: The number of rows in the sparse matrix (default: 30)
    :param L: The number of columns in the sparse matrix (default: 10)
    :param K: The signal sparsity (default: 7)
    :param sigma: The noise level in the measurement vector (default: 0.1)
    :param mu: The regularization parameter (default: 1.0)
    """
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
    adaptive, accelerated, plain = tests.test_modes(lambda **k: mmv(A, A.T, B, mu, X0, **k))
    plots.plot_convergence("Multiple Measurement Vector",
                           (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))

    # Plot the recovered signal
    plots.plot_matrices("Multiple Measurement Vector Recovery", X, adaptive[0])

    return adaptive, accelerated, plain

if __name__ == "__main__":
    test()
    plots.show_plots()
