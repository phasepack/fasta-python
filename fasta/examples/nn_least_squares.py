"""Solve the non-negative least squares problem, min_x .5||Ax-b||^2, x >= 0, using the FASTA solver."""

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

from fasta import fasta, plots, Convergence
from fasta.examples import ExampleProblem, test_modes
from fasta.operator import LinearOperator, Vector

__author__ = "Noah Singer"

__all__ = ["NNLeastSquaresProblem"]


class NNLeastSquaresProblem(ExampleProblem):
    def __init__(self, A: LinearOperator, At: LinearOperator, b: Matrix, x: Matrix):
        """Create an instance of the non-negative least squares problem.

        :param A: The measurement operator (must be linear, often simply a matrix)
        :param At: The Hermitian adjoint operator of A (for real matrices A, just the tranpose)
        :param b: The observation vector
        :param x: The true value of the unknown signal, if known (default: None)
        """
        super(ExampleProblem, self).__init__()

        self.A = A
        self.At = At
        self.b = b
        self.x = x

    def solve(self, x0: Matrix, fasta_options: float=None) -> Tuple[Vector, Convergence]:
        """Solve the non-negative least squares problem.

        :param x0: An initial guess for the solution
        :param fasta_options: Options for the FASTA algorithm (default: None)
        :return: The problem's computed solution and information on FASTA's convergence
        """
        f = lambda z: .5 * la.norm((z - self.b).ravel())**2
        gradf = lambda z: z - self.b
        g = lambda x: 0
        proxg = lambda x, t: np.maximum(x, 0)

        x = fasta(self.A, self.At, f, gradf, g, proxg, x0, **(fasta_options or {}))

        return x.solution, x

    @staticmethod
    def construct(M: int=200, N: int=1000, K: int=10, sigma: float=0.005) -> Tuple["NNLeastSquaresProblem", Convergence]:
        """Construct a sample non-negative least squares problem with a random sparse signal and measurement matrix.

        :param M: The number of measurements (default: 200)
        :param N: The dimension of the sparse signal (default: 1000)
        :param K: The signal sparsity (default: 10)
        :param sigma: The noise level in the observation vector (default: 0.005)
        :return: An example of this type of problem and a good initial guess for its solution
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

        return NNLeastSquaresProblem(A, A.T, b, x=x), x0

    def plot(self, solution: Vector) -> None:
        # Plot the recovered signal
        plots.plot_signals("Non-Negative Least Squares", self.x, solution)

if __name__ == "__main__":
    problem, x0 = NNLeastSquaresProblem.construct()
    print("Constructed non-negative least squares problem.")

    adaptive, accelerated, plain = test_modes(problem, x0)

    plots.plot_convergence("Non-Negative Least Squares", [adaptive[1], accelerated[1], plain[1]], ["Adaptive", "Accelerated", "Plain"])
    problem.plot(adaptive[0])
    plt.show()
