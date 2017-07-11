"""Solve the support vector machine problem,

min_w mu||w||^2 + 1/M*h(Dw,L),

where M is the number of observation vectors and the hinge loss function, h,
is defined as,

h(Z,L) = sum_i max(1 - l_i * z_i),

where l_i and z_i are the ith rows of Z and L, respectively.
The norm of w is minimized in order to promote a maximum-margin classifier.
"""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
import scipy.misc
from fasta import fasta, tests, proximal, plots


def svm(D, L, mu, Y0, **kwargs):
    """Solve the total variation denoising problem.

    :param M: A noisy image.
    :param mu: A parameter controlling the regularization.
    :param X0: An initial guess for the solution.
    :return: The output of the FASTA solver on the problem.
    """

    A = lambda y: D.T @ L @ y


    f = lambda Z: .5 * la.norm((Z - M/mu).ravel())**2
    gradf = lambda Z: Z - M/mu
    g = lambda Y: 0

    # Solve dual problem
    Y = fasta(div, grad, f, gradf, g, proxg, Y0, **kwargs)

    return M - mu * div(Y.solution), Y

if __name__ == "__main__":
    # Number of observation vectors
    M = 1000

    # Number of features per vector
    N = 15

    # Regularization parameter
    mu = 10

    # Mask representing (+) labels
    positive = np.random.permutation(M)[:M/2]

    # Generate linearly separable data
    D = np.random.randn(M,N) - 1
    D[positive] += 2

    # Generate labels
    L = -np.ones(M)
    L[positive] += 2

    # Initial iterate
    Y0 = np.zeros((N, N, 2))

    print("Constructed total-variation denoising problem.")

    # Test the three different algorithms
    plain, adaptive, accelerated = tests.test_modes(lambda **k: total_variation(M, mu, Y0, **k))

    # Plot the recovered signal
    plots.plot_images(M, adaptive[0])
    plots.show_plots()
