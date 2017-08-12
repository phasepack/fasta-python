"""Linear algebra routines used throughout FASTA and its related packages."""

import numpy as np
from scipy.sparse import linalg as sla
from typing import Callable, Tuple
from functools import reduce
from operator import mul

Matrix = np.ndarray
Vector = np.ndarray


class LinearMap:
    """A generalized linear map on n-dimensional arrays, which maps the vector space V to the vector space W.

    This linear map is some function, f, mapping from V to W, which satisfies the identity:

        f(a*x + b*y) = a*f(x) + b*f(y),

    for all vectors x, y in V and scalars a, b. This class improves scipy's LinearOperator, which only operates on
    single-dimensional arrays, to be able to operate on spaces with arbitrary rank.
    """
    def __init__(self, map_func: Callable[[np.ndarray], np.ndarray], adj_func: Callable[[np.ndarray], np.ndarray],
                 Vshape: Tuple[int, ...], Wshape: Tuple[int, ...]):
        """Create a linear map.

        :param op_func: A linear function from V to W.
        :param adj_func: The adjoint function to op_func, mapping from W to V.
        :param Vshape: The dimensions of ndarrays in V.
        :param Wshape: The dimensions of ndarrays in W.
        """
        self.map_func = map_func
        self.adj_func = adj_func
        self.Vshape = Vshape
        self.Wshape = Wshape

    @staticmethod
    def from_matrix(A: np.ndarray) -> "LinearMap":
        """Create a linear map from a 2D array."""
        assert A.ndim == 2
        return LinearMap(lambda x: A @ x, lambda x: A.T @ x, (A.shape[1],), (A.shape[0],))

    @staticmethod
    def identity(shape: Tuple[int, ...]) -> "LinearMap":
        """Create an identity linear operator.

        :param shape: The dimensions of ndarrays in V.
        :return: A map mapping every ndarray in V to itself.
        """
        return LinearMap(lambda x: x, lambda x: x, shape, shape)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """Linearly map a vector in V to its correspondingly vector in W.

        :param v: A vector v in V.
        :return: The image of v in W.
        """
        assert v.shape == self.Vshape
        w = self.map_func(v)
        assert w.shape == self.Wshape
        return w

    @property
    def H(self) -> "LinearMap":
        """Take the adjoint of this linear map.

        :return: The map adjoint to this linear map.
        """
        return LinearMap(self.adj_func, self.map_func, self.Wshape, self.Vshape)

    def __matmul__(self, B: "LinearMap") -> "LinearMap":
        """Compose this linear map with another linear map.

        :param B: A second linear map transforming the same spaces.
        :return: The composition of this map and B.
        """
        assert isinstance(B, LinearMap) and self.Wshape == B.Vshape
        return LinearMap(lambda x: self(B(x)), lambda x: B.H(self.H(x)), self.Vshape, B.Wshape)

    def __rmul__(self, k) -> "LinearMap":
        """Scale this linear map by a scalar.

        :param k: A scalar.
        :return: This map, scaled by k.
        """
        assert np.isscalar(k)
        return LinearMap(lambda x: k * self(x), lambda x: k * self.H(x), self.Vshape, self.Wshape)

    def __mul__(self, k) -> "LinearMap":
        """Scale this linear map by a scalar.

        :param k: A scalar.
        :return: This map, scaled by k.
        """
        return k * self

    def __neg__(self) -> "LinearMap":
        """Negate this linear map.

        :return: This linear map, negated."""
        return (-1) * self

    def __add__(self, B: "LinearMap") -> "LinearMap":
        """Add this linear map to another linear map.

        :param B: A second linear map transforming the same spaces.
        :return: The sum of this map and B.
        """
        assert isinstance(B, LinearMap) and self.Vshape == B.Vshape and self.Wshape == B.Wshape
        return LinearMap(lambda x: self(x) + B(x), lambda x: self.H(x) + B.H(x), self.Vshape, self.Wshape)

    def __sub__(self, B: "LinearMap") -> "LinearMap":
        """Subtract another linear map from this linear map.

        :param B: A second linear map transforming the same spaces.
        :return: The difference of this map and B.
        """
        return self + (-B)

    @property
    def is_operator(self) -> bool:
        """Check whether this linear map is an operator: whether V = W, so this linear map is an endomorphism on V."""
        return self.Vshape == self.Wshape

    def __pow__(self, n: int, modulo=None) -> "LinearMap":
        """Repeat this linear operator a number of times.

        This linear map must be a linear operator.

        :param n: A non-negative number of times to repeat this operator.
        :param modulo: Unused.
        :return: This operator, repeated n times.
        """
        assert self.is_operator

        new_map = LinearMap.identity(self.Vshape)
        for i in range(n):
            new_map @= self
        return new_map

    @property
    def _scipy(self) -> sla.LinearOperator:
        """Convert this linear map to a scipy LinearOperator in order to take advantage of scipy numerical routines."""
        M = reduce(mul, self.Vshape, 1)
        N = reduce(mul, self.Wshape, 1)

        return sla.LinearOperator((M,N), matvec=lambda x: np.ravel(self(x)), rmatvec=lambda x: np.ravel(self.H(x)))

    def eigs(self, k: int=1) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the eigenvalues and eigenvectors of this linear operator.

        This linear map must be a linear operator.

        :param k: The number of eigenvalue/eigenvector pairs to compute.
        :return: A tuple containing the eigenvalues and eigenvectors, respectively.
        """
        assert self.is_operator

        values, vectors = sla.eigs(self._scipy, k)
        return values, np.reshape(vectors, (k,)+self.Wshape)
