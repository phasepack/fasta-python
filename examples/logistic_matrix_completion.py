"""Solve the 1-bit matrix completion problem,

min_X mu||X||* + logit(X,B),

using the FASTA solver, where ||-||* denotes the sparse-inducing nuclear norm, and
the logistic log-odds function is defined as,

logit(Z,B) = sum_ij log(1 + e^(Z_ij)) - B_ij Z_ij."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots


def logistic_matrix_completion(B, mu, X0, **kwargs):
    """Solve the 1-bit matrix completion problem.

    :param B: A matrix of measurements.
    :param mu: A parameter controlling the regularization.
    :param X0: An initial guess for the solution.
    :param kwargs: Options for the FASTA solver.
    :return: The output of the FASTA solver on the problem.
    """

    # A and At are identities
    A = lambda X: X
    At = lambda X: X

    f = lambda Z: np.sum(np.log(1 + np.exp(Z)) - (B==1) * Z)
    gradf = lambda Z: -B / (1 + np.exp(B * Z))
    g = lambda X: mu * la.norm(np.diag(la.svd(X)[1]), 1)
    proxg = lambda X, t: proximal.project_Lnuc_ball(X, t*mu)

    return fasta(A, At, f, gradf, g, proxg, X0, **kwargs)

if __name__ == "__main__":
    # Number of rows
    M = 200

    # Number of colums
    N = 1000

    # Rank of matrix
    K = 10

    # Regularization parameter
    mu = 20

    # Create matrix and SVD factor it
    A = np.random.randn(M, N)
    U, s, V = la.svd(A)

    # Reduce the rank of s to K
    S = np.zeros((M, N))
    S[:K, :K] = np.diag(s[:K])

    # Reconstruct the matrix A, now with rank K
    A = U @ S @ V

    # Create observation vector
    P = 1 / (1 + np.exp(-A))
    B = 2.0 * (np.random.rand(M, N) < P) - 1

    # Initial iterate
    X0 = np.zeros((M, N))

    print("Constructed logistic matrix completion problem.")

    # Test the three different algorithms
    plain, adaptive, accelerated = tests.test_modes(lambda **k: logistic_matrix_completion(B, mu, X0, **k))

    # Plot the recovered signal
    plots.plot_matrices(B, adaptive.solution)
    plots.show_plots()