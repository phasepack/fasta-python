"""Test all the examples."""

from abc import ABCMeta, abstractclassmethod, abstractstaticmethod

TOLERANCE = 1E-3


class ExampleProblem(metaclass=ABCMeta):
    @abstractstaticmethod
    def construct(self):
        pass

    @abstractclassmethod
    def solve(self, initial_guess, fasta_options={ }):
        pass

    @abstractclassmethod
    def plot(self, solution):
        pass


def print_info(solution):
    print("Completed in {} iterations, {:f} seconds.".format(solution.iteration_count,
                                                             solution.times[solution.iteration_count] - solution.times[0]))


def test_modes(problem, x0):
    """Test the plain, adaptive, and accelerated modes of the FASTA algorithm on a given example problem."""

    print()

    print("Computing adaptive FBS.")
    adaptive = problem.solve(x0, {'tolerance': TOLERANCE, 'evaluate_objective': True, 'adaptive': True, 'accelerate': False})
    print_info(adaptive[1])

    print()

    print("Computing accelerated FBS.")
    accelerated = problem.solve(x0, {'tolerance': TOLERANCE, 'evaluate_objective': True, 'adaptive': False, 'accelerate': True})
    print_info(accelerated[1])

    print()

    print("Computing plain FBS.")
    plain = problem.solve(x0, {'tolerance': TOLERANCE, 'evaluate_objective': True, 'adaptive': False, 'accelerate': False})
    print_info(plain[1])

    print()

    return adaptive, accelerated, plain