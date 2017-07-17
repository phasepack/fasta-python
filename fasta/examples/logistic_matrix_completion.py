"""Solve the 1-bit matrix completion problem,

min_X mu||X||* + logit(X,B),

using the FASTA solver, where ||-||* denotes the sparsity-inducing nuclear norm, and
the logistic log-odds function is defined as,

logit(Z,B) = sum_ij log(1 + e^(Z_ij)) - B_ij * Z_ij."""

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots

__author__ = "Noah Singer"


def logistic_matrix_completion(B, mu, X0, **kwargs):
    """Solve the 1-bit matrix completion problem.

    :param B: A matrix of measurements
    :param mu: A parameter controlling the regularization
    :param X0: An initial guess for the solution
    :param kwargs: Options for the FASTA solver
    :return: The problem's computed solution and the full output of the FASTA solver on the problem
    """
    f = lambda Z: np.sum(np.log(1 + np.exp(Z)) - (B==1) * Z)
    gradf = lambda Z: -B / (1 + np.exp(B * Z))
    g = lambda X: mu * la.norm(np.diag(la.svd(X)[1]), 1)
    proxg = lambda X, t: proximal.project_Lnuc_ball(X, t*mu)

    X = fasta(None, None, f, gradf, g, proxg, X0, **kwargs)

    return X, X.solution


def test(M=200, N=1000, K=10, mu=20.0):
    """Construct a sample logistic matrix completion problem by factoring a random matrix, reducing its rank, and then randomly computing a logistic observation vector.

    :param M: The number of rows (default: 200)
    :param N: The number of columns (default: 1000)
    :param K: The rank of the reduced matrix (default: 10)
    :param mu: The regularization parameter (default: 20.0)
    """
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
    adaptive, accelerated, plain = tests.test_modes(lambda **k: logistic_matrix_completion(B, mu, X0, **k))
    plots.plot_convergence("Logistic Matrix Completion",
                           (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))

    # Plot the recovered signal
    plots.plot_matrices("Logistic Matrix Completion", B, adaptive[0])
    plots.show_plots()

if __name__ == "__main__":
    test()

del np, la
del fasta, tests, proximal, plots