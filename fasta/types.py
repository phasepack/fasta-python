"""Type hints used throughout the FASTA package."""

from typing import Callable, Union

Matrix = np.ndarray
Vector = np.ndarray
LinearOperator = Union[Callable[[np.ndarray], np.ndarray], Matrix, None]
