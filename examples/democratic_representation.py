"""Solve the democratic representation problem (L-inf-penalized least squares),

min_x mu||x||_inf + .5||Ax-b||^2,

using the FASTA solver. The solver promotes a solution with low dynamic range."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from scipy.fftpack import dct, idct
from fasta import fasta, tests, proximal, plots


def democratic_representation(A, At, b, mu, x0, **kwargs):
    """Solve the democratic representation problem.

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
    g = lambda x: mu * la.norm(x, np.inf)
    proxg = lambda x, t: proximal.project_Linf_ball(x, t*mu)

    x = fasta(A, At, f, gradf, g, proxg, x0, **kwargs)

    return x.solution, x


def test():
    # Number of measurements
    M = 500

    # Dimension of sparse signal
    N = 1000

    # Regularization parameter
    mu = 300

    # Choose a random set of DCT modes to sample
    samples = np.random.permutation(N - 1)[:M] + 1

    # Replace the last DCT mode with a 1, to force sampling the DC mode
    samples[M - 1] = 1

    # Sort the DCT modes
    samples.sort()

    # Create the subsampled DCT mask
    mask = np.zeros(N)
    mask[samples] = 1

    # Create matrix
    A = lambda x: mask * dct(x, norm='ortho')
    At = lambda x: idct(mask * x, norm='ortho')

    # Create random signal, where the unknown measurements correspond to the rows of the DCT that are sampled
    b = np.zeros(N)
    b[samples] = np.random.randn(M)

    # Initial iterate
    x0 = np.zeros(N)

    print("Constructed democratic representation problem.")

    # Test the three different algorithms
    plain, adaptive, accelerated = tests.test_modes(lambda **k: democratic_representation(A, At, b, mu, x0, **k))
    plots.plot_convergence("Democratic Representation",
                           (plain[1], adaptive[1], accelerated[1]), ("Plain", "Adaptive", "Accelerated"))

    # Plot the recovered signal
    plots.plot_signals("Democratic Representation", b, adaptive[0])
    plots.show_plots()

if __name__ == "__main__":
    test()
