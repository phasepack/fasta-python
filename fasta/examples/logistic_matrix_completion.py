"""Solve the 1-bit matrix completion problem, min_X mu||X||* + logit(X,B), using the FASTA solver.

||-||* denotes the sparsity-inducing nuclear norm, and the logistic log-odds function is defined as,

    logit(Z,B) = sum_ij log(1 + e^(Z_ij)) - B_ij * Z_ij."""

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

from fasta import fasta, proximal, plots
from fasta.examples import ExampleProblem, test_modes, NO_ARGS

__author__ = "Noah Singer"

__all__ = ["LogisticMatrixCompletionProblem"]


class LogisticMatrixCompletionProblem(ExampleProblem):
    def __init__(self, B, mu, X=None):
        """Create an instance of the logistic matrix completion problem.

        :param B: The observation matrix
        :param mu: The regularization parameter
        :param X: The true value of the unknown matrix, if known (default: None)
        """
        super(ExampleProblem, self).__init__()

        self.B = B
        self.mu = mu
        self.X = X

    @staticmethod
    def construct(M=200, N=1000, K=10, mu=20.0):
        """Construct a sample logistic matrix completion problem by factoring a random matrix, reducing its rank, and then randomly computing a logistic observation vector.

        :param M: The number of rows (default: 200)
        :param N: The number of columns (default: 1000)
        :param K: The rank of the reduced matrix (default: 10)
        :param mu: The regularization parameter (default: 20.0)
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

    def solve(self, X0, fasta_options=NO_ARGS):
        """Solve the 1-bit logistic matrix completion problem with FASTA.

        :param X0: An initial guess for the solution
        :param fasta_options: Additional options for the FASTA algorithm (default: None)"""
        f = lambda Z: np.sum(np.log(1 + np.exp(Z)) - (self.B == 1) * Z)
        gradf = lambda Z: -self.B / (1 + np.exp(self.B * Z))
        g = lambda X: self.mu * la.norm(np.diag(la.svd(X)[1]), 1)
        proxg = lambda X, t: proximal.project_Lnuc_ball(X, t*self.mu)

        X = fasta(None, None, f, gradf, g, proxg, X0, **fasta_options)

        return X.solution, X

    def plot(self, solution):
        plots.plot_matrices("Logistic Matrix Completion", self.X, solution)


if __name__ == "__main__":
    problem, X0 = LogisticMatrixCompletionProblem.construct()
    print("Constructed 1-bit logistic matrix completion problem.")

    adaptive, accelerated, plain = test_modes(problem, X0)

    plots.plot_convergence("Logistic Matrix Completion", (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))
    problem.plot(adaptive[0])
    plt.show()
