"""A collection of test harnesses, which compares different approaches (accelerative, adaptive, etc.) for a given problem,
and then plots the data and prints a report."""

__author__ = "Noah Singer"

import numpy as np
from matplotlib import pyplot as plt


def test_modes(solver, solution=None):
    print("Computing raw FBS...")
    raw = solver(accelerate=False, adaptive=False)

    print("Computing adaptive FBS...")
    adaptive = solver(accelerate=False, adaptive=True)

    print("Computing accelerated FBS...")
    accelerated = solver(accelerate=True, adaptive=False)

    plt.figure(1)
    for result in [raw, adaptive, accelerated]:
        plt.plot(np.log(result.residuals[:result.iteration_count]))

    plt.legend(("Raw", "Adaptive", "Accelerated"))
    plt.xlabel("Iteration #")
    plt.ylabel("log(residual)")
    plt.title("Convergence")

    if solution is not None:
        plt.figure(2)
        plt.plot(solution)
        for result in [raw, adaptive, accelerated]:
            plt.plot(result.solution)

        plt.legend(("Original", "Raw", "Adaptive", "Accelerated"))
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.title("Recovered Signals")

    plt.show()
