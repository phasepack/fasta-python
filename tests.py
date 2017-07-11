"""A collection of test harnesses, which solve a given problem in different ways."""

__author__ = "Noah Singer"

import numpy as np
from fasta import plots


TOLERANCE = 1E-8


def test_modes(solver):
    """Test the plain, adaptive, and accelerated modes of the FASTA algorithm."""
    print("Computing plain FBS...")
    plain = solver(accelerate=False, adaptive=False, evaluate_objective=True, tolerance=TOLERANCE)
    print("Completed in {} iterations.".format(plain[1].iteration_count))

    print()

    print("Computing adaptive FBS...")
    adaptive = solver(accelerate=False, adaptive=True, evaluate_objective=True, tolerance=TOLERANCE)
    print("Completed in {} iterations.".format(adaptive[1].iteration_count))

    print()

    print("Computing accelerated FBS...")
    accelerated = solver(accelerate=True, adaptive=False, evaluate_objective=True, tolerance=TOLERANCE)
    print("Completed in {} iterations.".format(accelerated[1].iteration_count))

    plots.plot_convergence((plain[1], adaptive[1], accelerated[1]), ("Plain", "Adaptive", "Accelerated"))

    return plain, adaptive, accelerated
