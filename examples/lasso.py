"""Solve the L1-penalized least-squares (lasso) problem,

min .5||Ax-b||^2, |x| < mu

using the FASTA solver. We express this as min f(Ax) + g(x), where f(Ax) = .5||Ax-b||^2
and g(x) = { 0           |x| < mu
           { infinity    otherwise."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, harness


def prox_infinity_norm(w, t):
    N = len(w)
    wabs = np.abs(w)

    alpha = max((np.cumsum(sorted(wabs, reverse=True)) - t) / np.arange(1, N+1))

    if alpha > 0:
        return min(min(wabs), alpha) * np.sign(w)
    else:
        return np.zeros(w.shape)


def lasso(A, x, b, mu, x0, **kwargs):
    f = lambda z: .5 * la.norm(z - b)**2
    gradf = lambda z: z - b
    g = lambda x: 0 if la.norm(g) < mu else np.inf

    # By Moreau's identity, we convert to proximal of conjugate problem (L-inf norm)
    proxg = lambda z, t: z - prox_infinity_norm(z, t)

    return fasta(A, A.T, f, gradf, g, proxg, x0, **kwargs)

if __name__ == "__main__":
    # Number of measurements
    M = 200

    # Dimension of spare signal
    N = 1000

    # Signal sparsity
    K = 10

    # Noise level in b
    sigma = 0.01

    # Create sparse signal
    x = np.zeros((N,1))
    x[np.random.permutation(N)[:K]] = 1

    # Regularization parameter
    mu = 0.8 * la.norm(x, 1)

    # Create matrix
    A = np.random.randn(M, N)
    A /= la.norm(A)

    # Create noisy observation vector
    b = A@x
    b += sigma * np.random.standard_normal(b.shape)

    # Initial iterate
    x0 = np.zeros((N, 1))

    print("Constructed lasso problem.")

    harness.test_modes(lambda **k: lasso(A, x, b, mu, x0, **k), solution=x)