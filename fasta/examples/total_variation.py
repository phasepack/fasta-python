"""Solve the total-variation denoising problem, min_X mu*TV(X) + .5*||X-M||^2, using the FASTA solver.

M is a noisy image and TV(X) represents the total-variation seminorm,

    TV(X) = sum_ij sqrt( (X_{i+1,j} - X_{i,j})^2 + (X_{i,j+1} - X_{i,j})^2 ).

This is accomplished by forming the dual problem,

    min_Y ||div(grad(Y)) - M/mu||^2.
"""

import numpy as np
from numpy import linalg as la
from scipy.misc import ascent
from fasta import fasta, tests, proximal, plots

__author__ = "Noah Singer"

__all__ = ["grad", "div", "total_variation", "test"]


def grad(X):
    """The gradient operator on an N-dimensional array, returning an (N+1)-dimensional array, where the
    (N+1)st dimension contains N entries, each representing the gradient in one direction.

    :param X: An N-dimensional array.
    :return: An (N+1)-dimensional array representing the discrete gradient of X.
    """
    # Allocate memory for gradient
    gradient = np.zeros(X.shape + (X.ndim,))

    for dim in range(X.ndim):
        # Set gradient to shifted matrix
        gradient[...,dim] = np.roll(X, 1, axis=dim) - X

    return gradient


def div(X):
    """The divergence operator on an N-dimensional array, returning an (N-1)-dimensional array. It computes
    backwards differences and sums those differences, acting as the adjoint operator to the gradient.

    :param X: An N-dimensional array.
    :return: An (N-1)-dimensional array representing the discrete divergence of X.
    """
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

    :param M: A noisy image
    :param mu: A parameter controlling the regularization
    :param Y0: An initial guess for the gradient of the solution
    :return: The problem's computed solution and the full output of the FASTA solver on the problem.
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


def test(sigma=0.05, mu=0.01):
    """Construct a sample total-variation denoising problem using the standard SciPy test image `ascent`.

    :param sigma: The noise level in the image (default: 0.05)
    :param mu: The regularization parameter (default: 0.01)
    """
    # Generate an image
    N = 512
    M = ascent().astype(float)

    # Normalize M
    M /= np.max(M)

    # Add noise
    M += sigma * np.random.randn(N, N)

    # Initial iterate
    Y0 = np.zeros((N, N, 2))

    print("Constructed total-variation denoising problem.")

    # Test the three different algorithms
    adaptive, accelerated, plain = tests.test_modes(lambda **k: total_variation(M, mu, Y0, **k))
    plots.plot_convergence("Total Variation Denoising",
                           (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))

    # Plot the recovered signal
    plots.plot_matrices("Total Variation Denoising", M, adaptive[0])

    return adaptive, accelerated, plain

if __name__ == "__main__":
    test()
    plots.show_plots()
