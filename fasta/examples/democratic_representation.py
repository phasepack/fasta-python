"""Solve the democratic representation problem (L-inf-penalized least squares), min_x mu||x||_inf + .5||Ax-b||^2, using the FASTA solver.

The solver promotes a solution with low dynamic range."""

import numpy as np
from numpy import linalg as la
from scipy.fftpack import dct, idct
from fasta import fasta, proximal, plots
from fasta.examples import ExampleProblem, test_modes

__author__ = "Noah Singer"

__all__ = ["democratic_representation", "test"]


class DemocraticRepresentationProblem(ExampleProblem):
    def __init__(self, A, At, b, mu, x=None):
        """Instantiate an instance of the democratic representation problem.

        :param A: A matrix or function handle
        :param At: The transpose of A
        :param b: A measurement vector
        :param mu: A parameter controlling the regularization
        :param x: The problem's true solution, if known (default: None)
        """
        super(ExampleProblem, self).__init__()

        self.A = A
        self.At = At
        self.b = b
        self.mu = mu
        self.x = x

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

        return DemocraticRepresentationProblem(A, At, b, mu, x=x), x0

if __name__ == "__main__":
    test()
    plots.show_plots()
