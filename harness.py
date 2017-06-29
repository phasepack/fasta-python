"""A collection of test harnesses, which compares different approaches (accelerative, adaptive, etc.) for a given problem,
and then plots the data and prints a report."""

__author__ = "Noah Singer"

import numpy as np
from matplotlib import pyplot as plt


def test_modes(solver, solution=None):
    print("Computing raw FBS...")
    raw = solver(accelerate=False, adaptive=False, evaluate_objective=True)
    print("Completed in {} iterations.".format(raw.iteration_count))

    print()

    print("Computing adaptive FBS...")
    adaptive = solver(accelerate=False, adaptive=True, evaluate_objective=True)
    print("Completed in {} iterations.".format(adaptive.iteration_count))

    print()

    print("Computing accelerated FBS...")
    accelerated = solver(accelerate=True, adaptive=False, evaluate_objective=True)
    print("Completed in {} iterations.".format(accelerated.iteration_count))

    plt.figure(1)

    plt.subplot(121)
    for result in [raw, adaptive, accelerated]:
        plt.plot(np.log(result.norm_residuals[:result.iteration_count]))

    plt.legend(("Raw", "Adaptive", "Accelerated"))
    plt.xlabel("Iteration #")
    plt.ylabel("log(norm residual)")
    plt.title("Normalized Residuals")

    plt.subplot(122)
    for result in [raw, adaptive, accelerated]:
        plt.plot(np.log(result.objectives[:result.iteration_count]))

    plt.legend(("Raw", "Adaptive", "Accelerated"))
    plt.xlabel("Iteration #")
    plt.ylabel("log(objective)")
    plt.title("Objective Function")

    if solution is not None:
        plt.figure(2)
        plt.plot(solution)
        plt.plot(adaptive.solution)

        plt.legend(("Original", "Recovered"))
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.title("Recovered Signals")

    plt.show()

    return raw, adaptive, accelerated
