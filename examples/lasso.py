"""Solve the L1-penalized least-squares problem,

min .5||Ax-b||^2, |x| < mu

using the FASTA solver. We express this as min f(Ax) + g(x), where f(Ax) = .5||Ax-b||^2
and g(x) = { 0           |x| < mu
           { infinity    otherwise."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots



def lasso(A, b, mu, x0, **kwargs):
    f = lambda z: .5 * la.norm(z - b)**2
    gradf = lambda z: z - b
    g = lambda z: 0
    proxg = lambda z, t: proximal.project_L1_ball(z, mu)

    return fasta(A, A.T, f, gradf, g, proxg, x0, **kwargs)

if __name__ == "__main__":
    # Number of measurements
    M = 200

    # Dimension of spare signal
    N = 1000

    # Signal sparsity
    K = 10

    # Noise level in b
    sigma = 0.01

    # Create sparse signal
    x = np.zeros(N)
    x[np.random.permutation(N)[:K]] = 1

    # Regularization parameter
    mu = 0.8 * la.norm(x, 1)

    # Create matrix
    A = np.random.randn(M, N)
    A /= la.norm(A, 2)

    # Create noisy observation vector
    b = A @ x + sigma * np.random.randn(M)

    # Initial iterate
    x0 = np.zeros(N)

    print("Constructed lasso problem.")

    # Test the three different algorithms
    raw, adaptive, accelerated = tests.test_modes(lambda **k: lasso(A, b, mu, x0, **k))

    # Plot the recovered signal
    plots.plot_signals(x, adaptive.solution)
    plots.show_plots()
