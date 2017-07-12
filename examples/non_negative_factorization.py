"""Solve the L1-penalized non-negative matrix factorization problem,

min_{X,Y} mu||X||_1 + ||S - XY^T||, X >= 0, Y >= 0, ||Y||_inf <= 1

using the FASTA solver. This problem is non-convex, but FBS is still often effective."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots


def non_negative_factorization(S, mu, X0, Y0, **kwargs):
    """Solve the L1-penalized non-negative matrix factorization problem.

    :param S: A matrix to factorize.
    :param mu: A parameter controlling the regularization.
    :param X0: An initial guess for the sparse factor.
    :param Y0: An initial guess for the small factor.
    :param kwargs: Options for the FASTA solver.
    :return: The output of the FASTA solver on the problem.
    """

    # Combine unknowns into single matrix so FASTA can handle them
    Z0 = np.concatenate((X0, Y0))

    # First N rows of Z are X, so X = Z[:N,...], Y = Z[N:,...]
    N = X0.shape[0]

    f = lambda Z: .5 * la.norm((S - Z[:N,...] @ Z[N:,...].T).ravel())**2

    def gradf(Z):
        X = Z[:N,...]
        Y = Z[N:,...]
        d = X @ Y.T - S
        return np.concatenate((d @ Y, d.T @ X))

    g = lambda Z: mu * la.norm(Z[:N,...].ravel(), 1)
    proxg = lambda Z, t: np.concatenate((proximal.shrink(Z[:N,...], t*mu), np.minimum(np.maximum(Z[N:,...], 0), 1)))

    Z = fasta(None, None, f, gradf, g, proxg, Z0, **kwargs)

    return (Z.solution[:N,...], Z.solution[N:,...]), Z

if __name__ == "__main__":
    # Rows of data matrix
    M = 800

    # Columns of data matrix
    N = 200

    # Rank of factorization
    K = 10

    # Regularization parameter
    mu = 1

    # Sparsity parameter for first factor
    b = 0.75

    # Noise level in observation matrix
    sigma = 0.1

    # Create random factor matrices
    X = np.random.rand(M, K)
    Y = np.random.rand(N, K)

    # Make X sparse
    X *= np.random.rand(M, K) > b

    # Create observation matrix
    S = X @ Y.T + sigma * np.random.randn(M, N)

    # Initial iterates
    X0 = np.zeros((M, K))
    Y0 = np.random.rand(N, K)

    print("Constructed non-negative matrix factorization problem.")

    # Test the three different algorithms
    plain, adaptive, accelerated = tests.test_modes(lambda **k: non_negative_factorization(S, mu, X0, Y0, **k))

    # Plot the recovered signal
    plots.plot_matrices("X", X, adaptive[0][0])
    plots.plot_matrices("Y", Y, adaptive[0][1])
    plots.show_plots()
