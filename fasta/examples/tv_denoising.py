"""Solve the total-variation denoising problem, min_X mu*TV(X) + .5*||X-M||^2, using the FASTA solver.

M is a noisy image and TV(X) represents the total-variation seminorm,

    TV(X) = sum_ij sqrt( (X_{i+1,j} - X_{i,j})^2 + (X_{i,j+1} - X_{i,j})^2 ).

This is accomplished by forming the dual problem,

    min_Y .5*||div(grad(Y)) - M/mu||^2.
"""

import numpy as np
from numpy import linalg as la
from scipy.misc import ascent
from matplotlib import pyplot as plt

from fasta import fasta, plots, Convergence
from fasta.examples import ExampleProblem, test_modes
from fasta.linalg import LinearOperator, Matrix

__author__ = "Noah Singer"

__all__ = ["grad", "div", "TVDenoisingProblem"]


def grad(X: Matrix) -> Matrix:
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


def div(X: Matrix) -> Matrix:
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


class TVDenoisingProblem(ExampleProblem):
    def __init__(self, M: Matrix, mu: float):
        """Create an instance of the total variation denoising problem.

        :param M: A noisy image
        :param mu: The regularization parameter
        """
        super(ExampleProblem, self).__init__()

        self.M = M
        self.mu = mu

    def solve(self, Y0: Matrix, fasta_options: dict=None) -> Tuple[Matrix, Convergence]:
        """Solve the total variation denoising problem.

        :param Y0: An initial guess for the gradient of the solution
        :param fasta_options: Options for the FASTA algorithm (default: None)
        :return: The problem's computed solution and convergence information on FASTA
        """
        f = lambda Z: .5 * la.norm((Z - self.M/self.mu).ravel())**2
        gradf = lambda Z: Z - self.M/self.mu
        g = lambda Y: 0

        def proxg(Y, t):
            # Norm of the gradient at each point in space
            norms = la.norm(Y, axis=Y.ndim-1)

            # Scale norms so that gradients have magnitude at least one
            norms = np.maximum(norms, 1)

            return Y / norms[...,np.newaxis]

        # Solve dual problem
        Y = fasta(div, grad, f, gradf, g, proxg, Y0, **(fasta_options or {}))

        X = self.M - self.mu * div(Y.solution)

        return X, Y

    @staticmethod
    def construct(sigma: float=0.1, mu: float=0.1) -> Tuple["TVDenoisingProblem", Matrix]:
        """Construct a sample total-variation denoising problem using the standard SciPy test image `ascent`.

        :param sigma: The noise level in the image (default: 0.1)
        :param mu: The regularization parameter (default: 0.01)
        :return: An example of this type of problem and a good initial guess for its solution
        """
        # Generate an image
        M = ascent().astype(float)

        # Normalize M
        M /= np.max(M)

        # Add noise
        M += sigma * np.random.randn(*M.shape)

        # Initial iterate
        Y0 = np.zeros(M.shape + (2,))

        return TVDenoisingProblem(M, mu), Y0

    def plot(self, solution: Matrix) -> None:
        """Plot the recovered, denoised image against the original noisy image.

        :param solution: The denoised image
        """
        plots.plot_matrices("Total Variation Denoising", self.M, solution)


if __name__ == "__main__":
    problem, Y0 = TVDenoisingProblem.construct()
    print("Constructed total-variation denoising problem.")

    adaptive, accelerated, plain = test_modes(problem, Y0)

    plots.plot_convergence("Total-Variation Denoising", [adaptive[1], accelerated[1], plain[1]], ["Adaptive", "Accelerated", "Plain"])
    problem.plot(adaptive[0])
    plt.show()
