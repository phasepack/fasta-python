"""Solve the democratic representation problem (L-inf-penalized least squares), min_x mu||x||_inf + .5||Ax-b||^2, using the FASTA solver.

The solver promotes a solution with low dynamic range."""

from typing import Tuple

import numpy as np
from fasta import fasta, proximal, plots
from fasta.examples import ExampleProblem, test_modes
from matplotlib import pyplot as plt
from numpy import linalg as la
from scipy.fftpack import dct, idct

from flow.linalg import LinearMap, Vector

__author__ = "Noah Singer"

__all__ = ["DemocraticRepresentationProblem"]


class DemocraticRepresentationProblem(ExampleProblem):
    def __init__(self, A: LinearMap, b: Vector, mu: float):
        """Instantiate an instance of the democratic representation problem.

        :param A: The measurement operator (must be linear, often simply a matrix)
        :param b: The observation vector
        :param mu: The regularization parameter
        """
        super(ExampleProblem, self).__init__()

        self.A = A
        self.b = b
        self.mu = mu

    def solve(self, x0: Vector, fasta_options: dict=None):
        """Solve the democratic representation problem.

        :param x0: An initial guess for the solution
        :param fasta_options: Options for the FASTA algorithm (default: None)
        :return: The computed democratic representation of the signal and information on FASTA's convergence
        """
        f = lambda z: .5 * la.norm((z - self.b).ravel()) ** 2
        gradf = lambda z: z - self.b
        g = lambda x: self.mu * la.norm(x, np.inf)
        proxg = lambda x, t: proximal.project_Linf_ball(x, t*self.mu)

        x = fasta(self.A, f, gradf, g, proxg, x0, **(fasta_options or {}))

        return x.solution, x

    @staticmethod
    def construct(M: int=500, N: int=1000, mu: float=300.0) -> Tuple["DemocraticRepresentationProblem", Vector]:
        """Construct a sample democratic representation problem with a randomly subsampled discrete cosine transform.

        :param M: The number of measurements (default: 500)
        :param N: The dimension of the sparse signal (default: 1000)
        :param mu: The regularization parameter (default: 300.0)
        :return: An example of this type of problem and a good initial guess for its solution
        """
        # Choose a random set of DCT modes to sample
        samples = np.random.permutation(N - 1)[:M] + 1

        # Replace the last DCT mode with a 1, to force sampling the DC mode
        samples[M - 1] = 1

        # Sort the DCT modes
        samples.sort()

        # Create the subsampled DCT mask
        mask = np.zeros(N)
        mask[samples] = 1

        # Create random signal, where the unknown measurements correspond to the rows of the DCT that are sampled
        b = np.zeros(N)
        b[samples] = np.random.randn(M)

        # Initial iterate
        x0 = np.zeros(N)

        return DemocraticRepresentationProblem(LinearMap(lambda x: mask * dct(x, norm='ortho'),
                                                              lambda x: idct(mask * x, norm='ortho'),
                                                              x0.shape, x0.shape), b, mu), x0

    def plot(self, solution: Vector) -> None:
        """Plot the computed democratic representation against the original signal.

        :param solution: The computed democratic representation
        """
        plots.plot_signals("Democratic Representation", self.b, solution)


if __name__ == "__main__":
    problem, x0 = DemocraticRepresentationProblem.construct()
    print("Constructed democratic representation problem.")

    adaptive, accelerated, plain = test_modes(problem, x0)

    plots.plot_convergence("Democratic Representation", [adaptive[1], accelerated[1], plain[1]], ["Adaptive", "Accelerated", "Plain"])
    problem.plot(adaptive[0])
    plt.show()
