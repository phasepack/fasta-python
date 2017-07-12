"""Solve the phase retrieval problem,

|a_i . x|^2 = b_i,

for some measurement vectors a_i and measured magnitudes b_i. This non-convex
problem is relaxed into the PhaseLift problem by letting A_i = a_i a_i^T and


using the FASTA solver. The problem is re-expressed with a characteristic function function for the constraint."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots


def lasso(A, At, b, mu, x0, **kwargs):
    """Solve the L1-restricted least squares problem.

    :param A: A matrix or function handle.
    :param At: The transpose of A.
    :param b: A measurement vector.
    :param mu: A parameter controlling the regularization.
    :param x0: An initial guess for the solution.
    :param kwargs: Options for the FASTA solver.
    :return: The output of the FASTA solver on the problem.
    """
    f = lambda z: .5 * la.norm((z - b).ravel())**2
    gradf = lambda z: z - b
    g = lambda x: 0 # TODO: add an extra condition to this
    proxg = lambda x, t: proximal.project_L1_ball(x, mu)

    x = fasta(A, At, f, gradf, g, proxg, x0, **kwargs)

    return x.solution, x

if __name__ == "__main__":
    # Number of measurements
    M = 500

    # Dimension of signal
    N = 100

    # Noise level in b
    sigma = 0.1

    # Create complex signal
    x = np.random.randn(N) + np.random.randn(N)*1j

    # Lifted representation
    X = x @ x.T

    # Create measurement matrix acting on column vector of lifted representation
    A = np.zeros((M, N**2))


    # Create matrix
    A = np.random.randn(M, N)
    A /= la.norm(A, 2)

    # Create noisy observation vector
    b = A @ x + sigma * np.random.randn(M)

    # Initial iterate
    x0 = np.zeros(N)

    print("Constructed lasso problem.")

    # Test the three different algorithms
    plain, adaptive, accelerated = tests.test_modes(lambda **k: lasso(A, A.T, b, mu, x0, **k))

    # Plot the recovered signal
    plots.plot_signals(x, adaptive.solution)
    plots.show_plots()
