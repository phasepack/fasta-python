"""A collection of test harnesses, which compares different approaches (accelerative, adaptive, etc.) for a given problem,
and then plots the data and prints a report."""

__author__ = "Noah Singer"

import numpy as np
from matplotlib import pyplot as plt


def test_modes(solver, solution=None):
    print("Computing raw FBS...")
    raw = solver(accelerate=False, adaptive=False, evaluate_objective=True)

    print("Computing adaptive FBS...")
    adaptive = solver(accelerate=False, adaptive=True, evaluate_objective=True)

    print("Computing accelerated FBS...")
    accelerated = solver(accelerate=True, adaptive=False, evaluate_objective=True)

    plt.figure(1)

    plt.subplot(221)
    for result in [raw, adaptive, accelerated]:
        plt.plot(np.log(result.residuals[:result.iteration_count]))

    plt.legend(("Raw", "Adaptive", "Accelerated"))
    plt.xlabel("Iteration #")
    plt.ylabel("log(residual)")
    plt.title("Residuals")

    plt.subplot(222)
    for result in [raw, adaptive, accelerated]:
        plt.plot(np.log(result.norm_residuals[:result.iteration_count]))

    plt.legend(("Raw", "Adaptive", "Accelerated"))
    plt.xlabel("Iteration #")
    plt.ylabel("log(norm residual)")
    plt.title("Normalized Residuals")

    plt.subplot(223)
    for result in [raw, adaptive, accelerated]:
        plt.plot(np.log(result.objectives[:result.iteration_count]))

    plt.legend(("Raw", "Adaptive", "Accelerated"))
    plt.xlabel("Iteration #")
    plt.ylabel("log(objective)")
    plt.title("Objective Function")

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

    return raw, adaptive, accelerated
