"""Solve the L1-penalized non-negative matrix factorization problem,

min_{X,Y} mu||X||_1 + ||S - XY^T||, X >= 0, Y >= 0, ||Y||_inf <= 1

using the FASTA solver. This problem is non-convex, but FBS is still often effective."""

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots

__author__ = "Noah Singer"


def non_negative_factorization(S, mu, X0, Y0, **kwargs):
    """Solve the L1-penalized non-negative matrix factorization problem.

    :param S: A matrix to factorize
    :param mu: A parameter controlling the regularization
    :param X0: An initial guess for the sparse factor
    :param Y0: An initial guess for the small factor
    :param kwargs: Options for the FASTA solver
    :return: The problem's computed solution and the full output of the FASTA solver on the problem.
    """
    # Combine unknowns into single matrix so FASTA can handle them
    Z0 = np.concatenate((X0, Y0))

    # First N rows of Z are X, so X = Z[:N,...], Y = Z[N:,...]
    N = X0.shape[0]

    f = lambda Z: .5 * la.norm((S - Z[:N,...] @ Z[N:,...].T).ravel())**2

    def gradf(Z):
        # Split the iterate matrix into the X and Y matrices
        X = Z[:N,...]
        Y = Z[N:,...]

        # Compute the actual gradient
        d = X @ Y.T - S
        return np.concatenate((d @ Y, d.T @ X))

    g = lambda Z: mu * la.norm(Z[:N,...].ravel(), 1)
    proxg = lambda Z, t: np.concatenate((proximal.shrink(Z[:N,...], t*mu), np.minimum(np.maximum(Z[N:,...], 0), 1)))

    Z = fasta(None, None, f, gradf, g, proxg, Z0, **kwargs)

    return (Z.solution[:N,...], Z.solution[N:,...]), Z


def test(M=800, N=200, K=10, b=0.75, sigma=0.1, mu=1.0):
    """Construct a sample non-negative factorization problem by computing two random matrices, making one sparse, and taking their product.

    :param M: The number of rows in the data matrix (default: 800)
    :param N: The number of columns in the data matrix (default: 200)
    :param K: The rank of the factorization (default: 10)
    :param b: The sparsity parameter for the first factor, X (default: 0.75)
    :param sigma: The noise level in the observation matrix (default: 0.1)
    :param mu: The regularization parameter (default: 1.0)
    """
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
    adaptive, accelerated, plain = tests.test_modes(lambda **k: non_negative_factorization(S, mu, X0, Y0, **k))
    plots.plot_convergence("Non-Negative Matrix Factorization",
                           (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))

    # Plot the recovered signal
    plots.plot_matrices("Factor X", X, adaptive[0][0])
    plots.plot_matrices("Factor Y", Y, adaptive[0][1])
    plots.show_plots()

if __name__ == "__main__":
    test()

del np, la
del fasta, tests, proximal, plots
