"""Solve the total-variation denoising problem,

min_X mu*TV(X) + .5*||X-M||^2,

using the FASTA solver, where M is a noisy image and TV(X) represents the total-variation seminorm,

TV(X) = sum_ij sqrt( (X_{i+1,j} - X_{i,j})^2 + (X_{i,j+1} - X_{i,j})^2 ).

This is accomplished by forming the dual problem,

min_Y ||div(grad(Y)) - M/mu||^2.
"""

__author__ = "Noah Singer"

import numpy as np
import scipy
from numpy import linalg as la
from fasta import fasta, tests, proximal, plots


def grad(X):
    """The gradient operator on an N-dimensional array, returning an (N+1)-dimensional array, where the
    (N+1)st dimension contains N entries, each representing the gradient in one direction."""

    # Allocate memory for gradient
    gradient = np.zeros(X.shape + (X.ndim,))

    for dim in range(X.ndim):
        # Set gradient to shifted matrix
        gradient[...,dim] = np.roll(X, 1, axis=dim) - X

    return gradient


def div(X):
    """The divergence operator on an N-dimensional array, returning an (N-1)-dimensional array. It performs
    backwards differences and sums the differences, acting as the adjoint operator to the gradient."""

    N = X.shape[-1]
    assert N == X.ndim-1

    # Allocate memory for divergence
    divergence = np.zeros(X.shape[:-1])

    for dim in range(N):
        # Take the partial derivative in X in our dimension
        dX = X[...,dim]

        # Shift backwards and add
        divergence += np.roll(dX, -1, axis=dim) - dX

    return divergence


def total_variation(M, mu, Y0, **kwargs):
    """Solve the total variation denoising problem.

    :param M: A noisy image.
    :param mu: A parameter controlling the regularization.
    :param Y0: An initial guess for the gradient of the solution.
    :return: The output of the FASTA solver on the problem.
    """

    f = lambda Z: .5 * la.norm((Z - M/mu).ravel())**2
    gradf = lambda Z: Z - M/mu
    g = lambda Y: 0

    def proxg(Y, t):
        # Norm of the gradient at each point in space
        norms = la.norm(Y, axis=Y.ndim-1)

        # Scale norms so that gradients have magnitude at least one
        norms = np.maximum(norms, 1)

        return Y / norms[...,np.newaxis]

    # Solve dual problem
    Y = fasta(div, grad, f, gradf, g, proxg, Y0, **kwargs)

    return M - mu * div(Y.solution), Y


def test():
    # Regularization parameter
    mu = 0.1

    # Noise level in M
    sigma = 0.05

    # Generate an image
    N = 512
    M = scipy.misc.ascent().astype(float)

    # Normalize M
    M /= np.max(M)

    # Add noise
    M += sigma * np.random.randn(N, N)

    # Initial iterate
    Y0 = np.zeros((N, N, 2))

    print("Constructed total-variation denoising problem.")

    # Test the three different algorithms
    plain, adaptive, accelerated = tests.test_modes(lambda **k: total_variation(M, mu, Y0, **k))

    # Plot the recovered signal
    plots.plot_matrices("Total Variation Denoising", M, adaptive[0])
    plots.show_plots()

if __name__ == "__main__":
    test()