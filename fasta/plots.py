"""A collection of plots of the results of FASTA."""

import numpy as np
from matplotlib import pyplot as plt

__author__ = "Noah Singer"


def plot_convergence(title, solvers, labels):
    """Plot the convergence curves of various solvers."""

    figure, (residuals, objective) = plt.subplots(1, 2)
    figure.suptitle("FASTA Convergence: {}".format(title))

    # Plot the normalized residuals
    residuals.set_xlabel("Iteration #")
    residuals.set_ylabel("log(norm residual)")
    residuals.set_title("Normalized Residuals")

    for solver, label in zip(solvers, labels):
        residuals.plot(np.log(solver.norm_residuals[:solver.iteration_count]), label=label)

    residuals.legend()

    # Plot the values of the objective function
    objective.set_xlabel("Iteration #")
    objective.set_ylabel("f(Ax) + g(x)")
    objective.set_title("Objective Function")

    for solver, label in zip(solvers, labels):
        objective.plot(solver.objectives[:solver.iteration_count], label=label)

    objective.legend()


def plot_signals(title, original, recovered):
    """Plot original and recovered signal vectors."""

    figure, axes = plt.subplots()
    figure.suptitle(title)

    axes.set_xlabel("Dimension")
    axes.set_ylabel("Index")
    axes.set_title("Recovered Signal")

    axes.plot(original, label="Original")
    axes.plot(recovered, label="Recovered")

    axes.legend()


def plot_matrices(title, original, recovered):
    """Plot original and recovered signal matrices."""

    min_entry = min(np.min(original), np.min(recovered))
    max_entry = max(np.max(original), np.max(recovered))

    figure, (original_axes, recovered_axes) = plt.subplots(1, 2)
    figure.suptitle(title)

    original_axes.set_title("Original")
    recovered_axes.set_title("Recovered")

    original_axes.matshow(original, vmin=min_entry, vmax=max_entry)
    im = recovered_axes.matshow(recovered, vmin=min_entry, vmax=max_entry)

    figure.colorbar(im, ax=(original_axes, recovered_axes))

    original_axes.set_aspect(original.shape[1] / original.shape[0])
    recovered_axes.set_aspect(recovered.shape[1] / recovered.shape[0])


def show_plots():
    """Display all plots."""

    plt.show()

del np, plt
