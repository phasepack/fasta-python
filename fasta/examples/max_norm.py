"""Solve the max-norm problem, min_X <S, X X^T>, ||X||_{2,inf}^2 <= 1 using the FASTA solver.

The NP-complete max-cut problem can be relaxed into this form. The inequality constrains the maximum L2-norm of any row
of X.
"""

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from fasta import fasta, plots
from fasta.examples import ExampleProblem, test_modes

__author__ = "Noah Singer"

__all__ = ["max_norm", "test"]

class MaxNormProblem(ExampleProblem):
    def __init__(self, S, mu):
        """Create an instance the max-norm problem.

        :param S: A square matrix
        :param mu: The regularization parameter
        """
        super(ExampleProblem, self).__init__()

        self.S = S
        self.mu = mu

def max_norm(S, mu, X0, **kwargs):
    """Solve the max-norm problem.

    :param S: A square matrix
    :param mu: A parameter controlling the regularization
    :param X0: An initial guess for the gradient of the solution
    :return: The problem's computed solution and the full output of the FASTA solver on the problem
    """
    f = lambda X: np.sum(S * (X @ X.T))
    gradf = lambda X: (S + S.T) @ X
    g = lambda X: 0

    def proxg(X, t):
        norms = la.norm(X, axis=1)

        # Shrink the norms that are too big, and ensure we don't divide by zero
        scale = np.maximum(norms, mu) + (norms == 0)

        return mu * X / scale[:,np.newaxis]

    X = fasta(None, None, f, gradf, g, proxg, X0, **kwargs)

    return X.solution, X


def test(N=2000, D=2, sigma=0.1, noise=0.15, dx=(1, 0.5), delta=0.01, K=10, mu=1.0):
    """Construct a sample max-norm problem by creating a two-moons segmentation dataset, converting it to a weighted graph, and then performing max-norm regularization on its adjacency matrix.

    :param N: The number of observations in the two-moons dataset (default: 2000)
    :param D: The dimensionality of the observation vectors (default: 2)
    :param sigma: The standard deviation of the similarity metric (default: 0.1)
    :param noise: The noise level in the two-moons points (default: 0.15)
    :param dx: The separation between the two moons in the x and y directions (default: (1, 0.5))
    :param delta: The balance parameter for the segmentation (default: 0.01)
    :param K: The maximum allowed rank of the factorization (default: 10)
    :param mu: The regularization parameter (default: 1.0)
    """
    # Points on a circle
    theta = np.arange(0, N) / N * 2 * np.pi

    # Embed circle in D dimensions
    points = np.zeros((N, D))
    points[:,0] = np.cos(theta)
    points[:,1] = np.sin(theta)

    # Separate out the top moon
    points[:N//2,:2] -= dx

    # Add noise
    points += noise * np.random.randn(N, D)

    # Build the similarity matrix from the distances between points
    distances = squareform(pdist(points))
    S = np.exp(-distances**2 / sigma**2 / 2)

    # The edge graph
    Q = delta - S

    # An initial guess for the iterate
    X0 = np.random.randn(N, K) / np.sqrt(K) / 10

    print("Constructed max-norm problem.")

    # Test the three different algorithms
    adaptive, accelerated, plain = tests.test_modes(lambda **k: max_norm(Q, mu, X0, **k))
    plots.plot_convergence("Max-Norm",
                           (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))

    labels = np.sign(adaptive[0] @ np.random.randn(K))

    figure, axes = plt.subplots()
    figure.suptitle("Max-Norm Optimization")

    axes.set_xlabel("Predicted value")
    axes.set_ylabel("Frequency")

    axes.plot(points[labels<0,0], points[labels<0,1], 'b.')
    axes.plot(points[labels>0,0], points[labels>0,1], 'r.')

    return adaptive, accelerated, plain

if __name__ == "__main__":
    test()
    plots.show_plots()
