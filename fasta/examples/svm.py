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
from fasta import fasta, tests, proximal, plots

__author__ = "Noah Singer"


def svm(D, L, C, y0, **kwargs):
    """Solve the support vector machine problem.

    :param D: The data matrix
    :param L: A vector of labels for the data
    :param Y0: An initial guess for the dual variable
    :return: The problem's computed solution and the full output of the FASTA solver on the problem.
    """
    f = lambda y: .5*la.norm((D.T @ (L * y)).ravel())**2 - np.sum(y)
    gradf = lambda y: L * (D @ (D.T @ (L * y))) - 1
    g = lambda y: 0
    proxg = lambda y, t: np.minimum(np.maximum(y, 0), C)

    # Solve dual problem
    y = fasta(None, None, f, gradf, g, proxg, y0, **kwargs)

    return D.T @ (L * y.solution), y


def test(M=1000, N=15, C=0.01):
    """Construct random linearly separable, labeled sample data for the SVM solver to classify.

    :param M: The number of observation vectors (default: 1000)
    :param N: The number of observed features per vector (default: 15)
    :param C: The regularization parameter (default: 0.01)
    """
    # Mask representing (+) and (-) labels
    permutation = np.random.permutation(M)
    negative = permutation[:M // 2]
    positive = permutation[M // 2:]

    # Generate linearly separable data
    D = 2 * np.random.randn(M, N)
    D[negative] -= 1.0
    D[positive] += 1.0

    # Generate labels
    L = np.zeros(M)
    L[negative] -= 1.0
    L[positive] += 1.0

    # Initial iterate
    y0 = np.zeros(M)

    print("Constructed support vector machine problem.")

    # Test the three different algorithms
    adaptive, accelerated, plain = tests.test_modes(lambda **k: svm(D, L, C, y0, **k))
    plots.plot_convergence("Support Vector Machine",
                           (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))

    w = adaptive[0]
    accuracy = np.sum(np.sign(D @ w) == L) / M

    # Plot a histogram of the residuals
    figure, axes = plt.subplots()
    figure.suptitle("Support Vector Machine (Accuracy: {}%)".format(accuracy * 100))

    axes.set_xlabel("Predicted value")
    axes.set_ylabel("Frequency")

    axes.hist((D[positive] @ w, D[negative] @ w), 25, label=("Positive", "Negative"))
    axes.legend()

    return adaptive, accelerated, plain

if __name__ == "__main__":
    test()
    plots.show_plots()
