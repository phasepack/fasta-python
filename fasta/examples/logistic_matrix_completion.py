"""Solve the 1-bit matrix completion problem, min_X mu||X||* + logit(X,B), using the FASTA solver.

||-||* denotes the sparsity-inducing nuclear norm, and the logistic log-odds function is defined as,

    logit(Z,B) = sum_ij log(1 + e^(Z_ij)) - B_ij * Z_ij."""

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
from typing import Tuple

from fasta import fasta, proximal, plots, Convergence
from fasta.examples import ExampleProblem, test_modes
from fasta.linalg import LinearOperator, Matrix

__author__ = "Noah Singer"

__all__ = ["LogisticMatrixCompletionProblem"]


class LogisticMatrixCompletionProblem(ExampleProblem):
    def __init__(self, B: Matrix, mu: float, X: Matrix=None):
        """Create an instance of the logistic matrix completion problem.

        :param B: The observation matrix
        :param mu: The regularization parameter
        :param X: The true value of the unknown matrix, if known (default: None)
        """
        super(ExampleProblem, self).__init__()

        self.B = B
        self.mu = mu
        self.X = X

    def solve(self, X0: Matrix, fasta_options: dict=None) -> Tuple[Matrix, Convergence]:
        """Solve the 1-bit logistic matrix completion problem with FASTA.

        :param X0: An initial guess for the solution
        :param fasta_options: Options for the FASTA algorithm (default: None)
        :return: The reconstructed matrix and information on FASTA's convergence
        """
        f = lambda Z: np.sum(np.log(1 + np.exp(Z)) - (self.B == 1) * Z)
        gradf = lambda Z: -self.B / (1 + np.exp(self.B * Z))
        g = lambda X: self.mu * la.norm(np.diag(la.svd(X)[1]), 1)
        proxg = lambda X, t: proximal.project_Lnuc_ball(X, t*self.mu)

        X = fasta(None, None, f, gradf, g, proxg, X0, **(fasta_options or {}))

        return X.solution, X

    @staticmethod
    def construct(M: int=200, N: int=1000, K: int=10, mu: float=20.0) -> Tuple["LogisticMatrixCompletionProblem", Matrix]:
        """Construct a sample logistic matrix completion problem by factoring a random matrix, reducing its rank, and then randomly computing a logistic observation vector.

        :param M: The number of rows (default: 200)
        :param N: The number of columns (default: 1000)
        :param K: The rank of the reduced matrix (default: 10)
        :param mu: The regularization parameter (default: 20.0)
        :return: An example of this type of problem and a good initial guess for its solution
        """
        # Create matrix and SVD factor it
        X = np.random.randn(M, N) * 10.0
        U, s, V = la.svd(X)

        # Reduce the rank of X to K by only taking the first K singular values
        S = np.zeros((M, N))
        S[:K, :K] = np.diag(s[:K])
        X = U @ S @ V

        # Create observation matrix
        P = 1 / (1 + np.exp(-X))
        B = 2.0 * (np.random.rand(M, N) < P) - 1

        # Initial iterate
        X0 = np.zeros((M, N))

        return LogisticMatrixCompletionProblem(B, mu, X=B), X0

    def plot(self, solution: Matrix) -> None:
        """Plot the recovered matrix against the original unknown matrix.

        :param solution: The recovered matrix
        """
        plots.plot_matrices("Logistic Matrix Completion", self.X, solution)


if __name__ == "__main__":
    problem, X0 = LogisticMatrixCompletionProblem.construct()
    print("Constructed 1-bit logistic matrix completion problem.")

    adaptive, accelerated, plain = test_modes(problem, X0)

    plots.plot_convergence("Logistic Matrix Completion", [adaptive[1], accelerated[1], plain[1]], ["Adaptive", "Accelerated", "Plain"])
    problem.plot(adaptive[0])
    plt.show()
