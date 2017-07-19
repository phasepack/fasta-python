"""Solve the L1-penalized least squares problem (also known as basis pursuit denoising, or BPDN), min_x mu||x||_1 + .5||Ax-b||^2, using the FASTA solver."""

import numpy as np
from numpy import linalg as la
from fasta import fasta, proximal, plots
from fasta.examples import ExampleProblem, test_modes

__author__ = "Noah Singer"

__all__ = ["sparse_least_squares", "test"]

class SparseLeastSquaresProblem(ExampleProblem):
    def __init__(self, A, At, b, mu, x=None):
        """Create an instance of the sparse least squares problem.

        :param A: The measurement operator (must be linear, often simply a matrix)
        :param At: The Hermitian adjoint operator of A (for real matrices A, just the transpose)
        :param b: The observation vector
        :param mu: The regularization parameter
        :param x: The true value of the unknown signal, if known (default: None)
        """
        super(SparseLeastSquares, self).__init__()

        self.A = A
        self.At = At
        self.b = b
        self.mu = mu
        self.x = x

def sparse_least_squares(A, At, b, mu, x0, **kwargs):
    """Solve the L1-penalized least squares problem.

    :param A: A matrix or function handle
    :param At: The transpose of A
    :param b: A measurement vector
    :param mu: A parameter controlling the regularization
    :param x0: An initial guess for the solution
    :param kwargs: Options for the FASTA solver
    :return: The problem's computed solution and the full output of the FASTA solver on the problem.
    """
    f = lambda z: .5 * la.norm((z - b).ravel())**2
    gradf = lambda z: z - b
    g = lambda x: mu * la.norm(x.ravel(), 1)
    proxg = lambda x, t: proximal.shrink(x, t*mu)

    x = fasta(A, At, f, gradf, g, proxg, x0, **kwargs)

    return x.solution, x


def test(M=200, N=1000, K=10, sigma=0.01, mu=0.02):
    """Construct a sample sparse least squares with a random sparse signal and coefficient matrix.

    :param M: The number of measurements (default: 200)
    :param N: The dimension of the sparse signal (default: 1000)
    :param K: The signal sparsity (default: 10)
    :param sigma: The noise level in the observation vector (default: 0.01)
    :param mu: The regularization parameter (default: 0.02)
    """
    # Create sparse signal
    x = np.zeros(N)
    x[np.random.permutation(N)[:K]] = 1

    # Create matrix
    A = np.random.randn(M, N)
    A /= la.norm(A, 2)

    # Create noisy observation vector
    b = A @ x + sigma * np.random.randn(M)

    # Initial iterate
    x0 = np.zeros(N)

    print("Constructed sparse least squares problem.")

    # Test the three different algorithms
    adaptive, accelerated, plain = tests.test_modes(lambda **k: sparse_least_squares(A, A.T, b, mu, x0, **k))
    plots.plot_convergence("Sparse Least Squares",
                           (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))

    # Plot the recovered signal
    plots.plot_signals("Sparse Least Squares Regression", x, adaptive[0])

    return adaptive, accelerated, plain

if __name__ == "__main__":
    test()
    plots.show_plots()
