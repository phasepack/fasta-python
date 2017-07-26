"""Solve the democratic representation problem (L-inf-penalized least squares), min_x mu||x||_inf + .5||Ax-b||^2, using the FASTA solver.

The solver promotes a solution with low dynamic range."""

import numpy as np
from numpy import linalg as la
from scipy.fftpack import dct, idct
from matplotlib import pyplot as plt

from fasta import fasta, proximal, plots
from fasta.examples import ExampleProblem, test_modes, NO_ARGS

__author__ = "Noah Singer"

__all__ = ["DemocraticRepresentationProblem"]


class DemocraticRepresentationProblem(ExampleProblem):
    def __init__(self, A, At, b, mu):
        """Instantiate an instance of the democratic representation problem.

        :param A: A matrix or function handle
        :param At: The transpose of A
        :param b: A measurement vector
        :param mu: A parameter controlling the regularization
        """
        super(ExampleProblem, self).__init__()

        self.A = A
        self.At = At
        self.b = b
        self.mu = mu

    @staticmethod
    def construct(M=500, N=1000, mu=300.0):
        """Construct a sample democratic representation problem with a randomly subsampled discrete cosine transform.

        :param M: The number of measurements (default: 500)
        :param N: The dimension of the sparse signal (default: 1000)
        :param mu: The regularization parameter (default: 300.0)
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

        # Create matrix
        A = lambda x: mask * dct(x, norm='ortho')
        At = lambda x: idct(mask * x, norm='ortho')

        # Create random signal, where the unknown measurements correspond to the rows of the DCT that are sampled
        b = np.zeros(N)
        b[samples] = np.random.randn(M)

        # Initial iterate
        x0 = np.zeros(N)

        return DemocraticRepresentationProblem(A, At, b, mu), x0

    def solve(self, x0, fasta_options=NO_ARGS):
        """Solve the democratic representation problem.
        :param A: A matrix or function handle.
        :param At: The transpose of A.
        :param b: A measurement vector.
        :param mu: A parameter controlling the regularization.
        :param x0: An initial guess for the solution.
        :param kwargs: Options for the FASTA solver.
        :return: The output of the FASTA solver on the problem.
        """
        f = lambda z: .5 * la.norm((z - self.b).ravel()) ** 2
        gradf = lambda z: z - self.b
        g = lambda x: self.mu * la.norm(x, np.inf)
        proxg = lambda x, t: proximal.project_Linf_ball(x, t*self.mu)

        x = fasta(self.A, self.At, f, gradf, g, proxg, x0, **fasta_options)

        return x.solution, x

    def plot(self, solution):
        plots.plot_signals("Democratic Representation", self.b, solution)


if __name__ == "__main__":
    problem, x0 = DemocraticRepresentationProblem.construct()
    print("Constructed democratic representation problem.")

    adaptive, accelerated, plain = test_modes(problem, x0)

    plots.plot_convergence("Democratic Representation", (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))
    problem.plot(adaptive[0])
    plt.show()
