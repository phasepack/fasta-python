"""Solve the support vector machine problem, min_w ||w||^2 + C*h(Dw,L), using the FASTA solver.

The hinge loss function, h, is defined as,

    h(Z,L) = sum_i max(1 - l_i * z_i),

where l_i and z_i are the ith rows of Z and L, respectively. The norm of w is minimized in order to promote a
maximum-margin classifier. The problem is solved by formulating the dual problem,

    min_y .5*||D^T L y||^2 - sum(y).
"""

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

from fasta import fasta, plots
from fasta.examples import ExampleProblem, test_modes
from fasta.operator import LinearOperator, Vector, Matrix

__author__ = "Noah Singer"

__all__ = ["generate", "SVMProblem"]


def generate(M: int, N: int, w: Vector) -> Matrix:
    """Generate linearly separable data."""
    # Mask representing (+) and (-) labels
    permutation = np.random.permutation(M)
    negative = permutation[:M // 2]
    positive = permutation[M // 2:]

    # Generate linearly separable data
    D = 2 * np.random.randn(M, N)
    D[negative] -= w
    D[positive] += w

    # Generate labels
    L = np.zeros(M)
    L[negative] -= 1.0
    L[positive] += 1.0

    return D, L


class SVMProblem(ExampleProblem):
    def __init__(self, D: Matrix, l: Vector, C, w=None):
        """Create an instance of the SVM classification problem.

        :param D: The data matrix
        :param l: A vector of labels for the data
        :param C: The regularization parameter
        """
        super(ExampleProblem, self).__init__()

        self.D = D
        self.l = l
        self.C = C
        self.w = w

    def solve(self, y0: Vector, fasta_options: dict=None) -> Tuple[Vector, Convergence]:
        """Solve the support vector machine problem.

        :param Y0: An initial guess for the dual variable
        :param fasta_options: Options for the FASTA algorithm (default: None)
        :return: The computing hyperplane separating the data and information on FASTA's convergence
        """
        f = lambda y: .5* la.norm((self.D.T @ (self.l * y)).ravel()) ** 2 - np.sum(y)
        gradf = lambda y: self.l * (self.D @ (self.D.T @ (self.l * y))) - 1
        g = lambda y: 0
        proxg = lambda y, t: np.minimum(np.maximum(y, 0), self.C)

        # Solve dual problem
        y = fasta(None, None, f, gradf, g, proxg, y0, **(fasta_options or {}))

        x = self.D.T @ (self.l * y.solution)

        return x, y

    @staticmethod
    def construct(M: int=1000, N: int=15, C: float=0.01, separation: float=1.0) -> Tuple["SVMProblem", Vector]:
        """Construct random linearly separable, labeled sample training data for the SVM solver to train on.

        :param M: The number of observation vectors (default: 1000)
        :param N: The number of observed features per vector (default: 15)
        :param C: The regularization parameter (default: 0.01)
        :param separation: The distance to move the data from the generated hyperplane (default: 1.0)
        :return: An example of this type of problem and a good initial guess for its solution
        """
        # Hyperplane separating the data
        w = np.random.randn(N)
        w /= la.norm(w)
        w *= separation

        D, l = generate(M, N, w)

        # Initial iterate
        y0 = np.zeros(M)

        return SVMProblem(D, l, C, w=w), y0

    def plot(self, solution: Vector, M_train: int=300, hist_size: int=25) -> None:
        """Plot the results of the computed SVM classifier on randomly generated linearly separable data.

        :param solution: The computed separating hyperplane
        :param M_train: The size of the test dataset
        :param hist_size: The number of bars in the visualization histogram
        """
        N = solution.shape[0]
        D_train, l_train = generate(M_train, N, self.w)

        accuracy = np.sum(np.sign(D_train @ solution) == l_train) / M_train

        # Plot a histogram of the residuals
        figure, axes = plt.subplots()
        figure.suptitle("Support Vector Machine (Accuracy: {}%)".format(accuracy * 100))

        axes.set_xlabel("Predicted value")
        axes.set_ylabel("Frequency")

        axes.hist((D_train[l_train == 1] @ solution, D_train[l_train == -1] @ solution),
                  hist_size, label=("Positive", "Negative"))
        axes.legend()


if __name__ == "__main__":
    problem, y0 = SVMProblem.construct()
    print("Constructed support vector machine problem.")

    adaptive, accelerated, plain = test_modes(problem, y0)

    plots.plot_convergence("Support Vector Machine", [adaptive[1], accelerated[1], plain[1]], ["Adaptive", "Accelerated", "Plain"])
    problem.plot(adaptive[0])
    plt.show()
