"""A test harness for FASTA, which compares different approaches (accelerative, adaptive, etc.) for a given problem,
and then plots the data and prints a report."""

__author__ = "Noah Singer"

import numpy as np
from matplotlib import pyplot as plt


def harness(results, labels):
    for result, label in zip(results, labels):
        plt.plot(np.log(result.residuals[:result.iteration_count]))

    plt.legend(labels)
    plt.xlabel("Iteration #")
    plt.ylabel("log(Residual)")
    plt.title("Convergence")
    plt.show()