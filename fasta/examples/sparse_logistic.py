"""Solve the L1-penalized logistic least squares problem, min mu||x||_1 + logit(Ax,b), using the FASTA solver.

The logistic log-odds function is defined as,

    logit(z,b) = sum_i log(1 + e^(z_i)) - b_i * z_i,

where z_i and b_i are the ith rows of z and b, respectively."""

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

from fasta import fasta, proximal, plots
from fasta.examples import ExampleProblem, test_modes

__author__ = "Noah Singer"

__all__ = ["SparseLogisticProblem"]


class SparseLogisticProblem(ExampleProblem):
    def __init__(self, A, At, b, mu, x):
        """Create an instance of the sparse logistic least squares problem.

        :param A: The measurement operator (must be linear, often simply a matrix)
        :param At: The Hermitian adjoint operator of A (for real matrices A, just the transpose)
        :param b: The observation vector
        :param mu: The regularization parameter
        :param x: The true value of the unknown signal, if known (default: None)
        """
        super(ExampleProblem, self).__init__()

        self.A = A
        self.At = At
        self.b = b
        self.mu = mu
        self.x = x

    @staticmethod
    def construct(M=1000, N=2000, K=5, mu=40):
        """Construct a sample sparse logistic least squares problem with a random sparse signal and measurement matrix.

        :param M: The number of measurements (default: 1000)
        :param N: The dimension of the sparse signal (default: 2000)
        :param K: The signal sparsity (default: 5)
        :param mu: The regularization parameter (default: 40.0)
        :return: An example of this type of problem and a good initial guess for its solution
        """
        # Create sparse signal
        x = np.zeros(N)
        x[np.random.permutation(N)[:K]] = 1

        # Create matrix
        A = np.random.randn(M, N)

        # Create observation vector
        p = 1 / (1 + np.exp(-A @ x))
        b = 2.0 * (np.random.rand(M) < p) - 1

        # Initial iterate
        x0 = np.zeros(N)

        return SparseLogisticProblem(A, A.T, b, mu, x=x), x0

    def solve(self, x0, fasta_options=None):
        """Solve the L1-penalized logistic least squares problem.

        :param x0: An initial guess for the solution
        :param fasta_options: Options for the FASTA algorithm (default: None)
        :return: The problem's computed solution and convergence information on FASTA
        """
        f = lambda z: np.sum(np.log(1 + np.exp(z)) - (self.b==1) * z)
        gradf = lambda z: -self.b / (1 + np.exp(self.b * z))
        g = lambda x: self.mu * la.norm(x.ravel(), 1)
        proxg = lambda x, t: proximal.shrink(x, t*self.mu)

        x = fasta(self.A, self.At, f, gradf, g, proxg, x0, **(fasta_options or {}))

        return x.solution, x

    def plot(self, solution):
        plots.plot_signals("Sparse Logistic Least Squares", self.x, solution)


if __name__ == "__main__":
    problem, x0 = SparseLogisticProblem.construct()
    print("Constructed sparse logistic least squares problem.")

    adaptive, accelerated, plain = test_modes(problem, x0)

    plots.plot_convergence("Sparse Logistic Least Squares", [adaptive[1], accelerated[1], plain[1]], ["Adaptive", "Accelerated", "Plain"])
    problem.plot(adaptive[0])
    plt.show()
