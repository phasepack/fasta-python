"""A collection of test harnesses, which solve a given problem in different ways using FASTA."""

__author__ = "Noah Singer"

TOLERANCE = 1E-8


def test_modes(solver):
    """Test the plain, adaptive, and accelerated modes of the FASTA algorithm."""
    print("Computing adaptive FBS.")
    adaptive = solver(accelerate=False, adaptive=True, evaluate_objective=True, tolerance=TOLERANCE)
    print("Completed in {} iterations.".format(adaptive[1].iteration_count))

    print()

    print("Computing accelerated FBS.")
    accelerated = solver(accelerate=True, adaptive=False, evaluate_objective=True, tolerance=TOLERANCE)
    print("Completed in {} iterations.".format(accelerated[1].iteration_count))

    print()

    print("Computing plain FBS.")
    plain = solver(accelerate=False, adaptive=False, evaluate_objective=True, tolerance=TOLERANCE)
    print("Completed in {} iterations.".format(plain[1].iteration_count))

    return adaptive, accelerated, plain
