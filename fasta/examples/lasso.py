"""Solve the L1-restricted least squares problem (also known as the LASSO problem), min_x .5||Ax-b||^2, ||x||_1 < mu, using the FASTA solver.

The problem is re-expressed with a characteristic function function for the constraint."""

import numpy as np
from numpy import linalg as la
from fasta import fasta, proximal, plots
from fasta.examples import ExampleProblem, test_modes

__author__ = "Noah Singer"

__all__ = ["lasso", "test"]


#TODO: document return values

class LASSOProblem(ExampleProblem):
    def __init__(self, A, At, b, mu, x=None):
        """Instantiate an instance of the LASSO problem.

        :param A: A matrix or function handle
        :param At: The transpose of A
        :param b: A measurement vector
        :param mu: A parameter controlling the regularization
        :param x: The true solution to the problem, if known (default: None)
        """
        super(ExampleProblem, self).__init__()

        self.A = A
        self.At = At
        self.b = b
        self.mu = mu
        self.x = x

    @staticmethod
    def construct(M=200, N=1000, K=10, sigma=0.01, mu=0.8):
        """Construct a sample LASSO regression problem with a random sparse signal and measurement matrix.

        :param M: The number of measurements (default: 200)
        :param N: The dimension of the sparse signal (default: 1000)
        :param K: The signal sparsity (default: 10)
        :param sigma: The noise level in the observation vector (default: 0.01)
        :param mu: The regularization parameter (default: 0.8)
        """
        # Create sparse signal
        x = np.zeros(N)
        x[np.random.permutation(N)[:K]] = 1

        # Normalize the regularization parameter
        mu *= la.norm(x, 1)

        # Create matrix
        A = np.random.randn(M, N)
        A /= la.norm(A, 2)

        # Create noisy observation vector
        b = A @ x + sigma * np.random.randn(M)

        # Initial iterate
        x0 = np.zeros(N)

        return LASSOProblem(A, A.T, b, mu, x=x), x0

    def solve(self, x0, fasta_options=None):
        """Solve the LASSO regression problem with FASTA.

        :param x0: An initial guess for the solution
        :param fasta_options: Additional options for the FASTA algorithm (default: None)
        """
        f = lambda z: .5 * la.norm((z - self.b).ravel()) ** 2
        gradf = lambda z: z - self.b
        g = lambda x: 0  # TODO: add an extra condition to this
        proxg = lambda x, t: proximal.project_L1_ball(x, self.mu)

        if not fasta_options:
            fasta_options = {}

        x = fasta(self.A, self.At, f, gradf, g, proxg, x0, **fasta_options)

        return x.solution, x

    def plot(self, solution):
        plots.plot_signals("LASSO", self.x, solution)


if __name__ == "__main__":
    problem, x0 = LASSOProblem.construct()

    adaptive, accelerated, plain = test_modes(problem, x0)

    plots.plot_convergence("LASSO", (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))
    problem.plot(adaptive[0])
    plots.show_plots()

