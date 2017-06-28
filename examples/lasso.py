"""Solve the L1-penalized least-squares problem,

min .5||Ax-b||^2, |x| < mu

using the FASTA solver. We express this as min f(Ax) + g(x), where f(Ax) = .5||Ax-b||^2
and g(x) = { 0           |x| < mu
           { infinity    otherwise."""

__author__ = "Noah Singer"

import numpy as np
from numpy import linalg as la
from fasta import fasta, harness


def prox_infinity_norm(w, t):
    wabs = np.abs(w)
    flipped = np.flip(np.sort(wabs, axis=0), axis=0)

    alpha = np.max((np.cumsum(flipped) - t) / np.arange(1, len(w) + 1))

    if alpha > 0:
        return np.minimum(wabs, alpha) * np.sign(w)
    else:
        return np.zeros(w.shape)


def lasso(A, b, mu, x0, **kwargs):
    f = lambda z: .5 * la.norm(z - b)**2
    gradf = lambda z: z - b
    g = lambda z: 0

    # By Moreau's identity, we convert to proximal of conjugate problem (L-inf norm)
    proxg = lambda z, t: z - prox_infinity_norm(z, mu)

    return fasta(A, A.T, f, gradf, g, proxg, x0, **kwargs)

if __name__ == "__main__":
    # np.random.seed(13409823)

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
    A /= la.norm(A, 1)

    # Create noisy observation vector
    b = A @ x
    b += sigma * np.random.standard_normal(b.shape)

    # Initial iterate
    x0 = np.zeros((N, 1))

    print("Constructed lasso problem.")

    raw, adaptive, accelerated = harness.test_modes(lambda **k: lasso(A, b, mu, x0, **k), solution=x)