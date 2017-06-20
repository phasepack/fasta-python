"""A Python implementation of Goldstein et. al's Fast Adaptive Shrinkage/Thresholding Algorithm.

FASTA minimizes a function expressed in the form
        min h(x) = f(Ax) + g(x)
where `f` is convex and differentiable and `g` is convex.
`g` may be non-differentiable, or possibly not even finite-valued, so normal gradient descent methods cannot be used.

The FASTA algorithm is a variant of the Forward-Backward Splitting (FBS) algorithm, which alternatively takes steps
to optimize `h` by optimizing `f` and then `g`. The optimization of `g` relies on the computation of the proximal
operator of `g` with stepsize `t`, equal to argmin { t*g(x) + .5 ||x-z||**2 }, which finds a point `x` close to the
minimum value of `g` while not straying too far from `z`.

The FASTA algorithm incorporates various improvements to FBS, as detailed in (Goldstein et al. 2016), including
adaptive stepsize selection, acceleration, preconditioning, non-monotone backtracking line search, continuation,
and a variety of stopping conditions.
"""

__author__ = "Noah Singer"


from .utils import functionize
from .stopping import hybrid_residual
import numpy as np

EPSILON = 1E-12


# TODO: check mutually allowed modes

def fasta(A, At, f, gradf, g, proxg, x0,
          max_iters=10000,
          tolerance=1E-3,
          stop_rule=hybrid_residual,

          L=None,
          tau0=None,

          backtrack=False,
          stepsize_shrink=0.2,
          window=10,
          max_backtracks=20,

          adaptive=True,
          accelerate=False,
          restart=True,
          evaluate_objective=False):
    """

    :param A: A matrix, or a function that returns a matrix, which is equal to A*x.
    :param At: The adjoint (conjugate transpose) of A.
    :param f: A convex, differentiable function of x.
    :param gradf: The gradient of f.
    :param g: A convex function of x.
    :param proxg: The proximal operator of g with stepsize t.
    :param x0: An initial guess for position of the optimal value (often a vector of zeroes).
    :param max_iters: The maximum number of iterations allowed by the algorithm (default: 1000).
    :param tolerance: When the residuals begin to decrease by less than this amount, the algorithm terminates (default: 1e-3).
    :param stop_rule: A function that checks whether the algorithm should terminate (default: stopping.hybrid_residual).
    :param L: The Lipschitz constant of f (default: the term is approximated). Only required if tau is not set.
    :param tau: The initial stepsize for the algorithm (default: computed from L).
    :param backtrack: Use backtracking line search (default: True).
    :param stepsize_shrink: When backtracking, aggressively decrease the step size to prevent further mistakes (default: 0.2).
    :param window: The lookback window for backtracking (default: 10).
    :param max_backtracks: The maximum total number of backtracks allowed in a single iteration of the algorithm (default: 20).
    :param adaptive: Adaptively choose the stepsize by locally approximating the function as a quadratic (default: True).
    :param accelerate: Increase the stepsize at every step of the algorithm (default: False).
    :param restart: Restart the acceleration of FISTA. Only relevant when accelerating (default: True).
    :return: A guess at an optimizer of h.
    """

    # Conventions:
    #   - Variables ending with 0 indicate the previous round's values
    #   - Variables ending with 1 indicate the current round's values
    #   - Variables ending with _hist indicate a history that is tracked between rounds

    A = functionize(A)
    At = functionize(At)

    # Check if we need to approximate the Lipschitz constant of f
    if not L or not tau0:
        # Compute len(x0) pairs of points
        x1 = np.random.randn(len(x0))
        x2 = np.random.randn(len(x1))

        # Compute the gradients at each pair of points
        gradf1 = At(gradf(A(x1)))
        gradf2 = At(gradf(A(x2)))

        # Approximate the Lipschitz constant of f
        L = np.max(np.norm(gradf1 - gradf2) / np.norm(x1 - x2))

        # We're guaranteed that FBS converges for tau <= 2.0 / L
        tau0 = (2 / L) / 10

    if not tau0:
        tau0 = 1 / L
    else:
        L = 1 / tau0

    # Allocate memory for convergence information
    residual_hist = np.zeros(max_iters)
    norm_residual_hist = np.zeros(max_iters)
    tau_hist = np.zeros(max_iters)
    f_hist = np.zeros(max_iters)

    if evaluate_objective:
        objective_hist = np.zeros(max_iters)

    total_backtracks = 0

    # Initialize values
    x1 = x0
    tau1 = tau0

    d1 = A(x1)
    f1 = f(d1)
    gradf1 = At(gradf(x1))
    f_hist[0] = f1

    # Additional initialization for accelerative descent
    if accelerate:
        x_accel1 = x1
        d_accel1 = d1
        alpha1 = 1

    # Additional initialization for backtracking
    if backtrack:
        total_backtracks = 0

    # Non-monotonicity for stopping conditions
    max_residual = -np.inf
    min_objective = np.inf

    # Algorithm loop
    i = 1
    while i <= max_iters:
        # Rename last iteration's current variables to this round's former variables
        x0 = x1
        gradf0 = gradf1
        tau0 = tau1

        # Perform FBS: Take the forwards step by moving x in the direction of f's gradient
        x1hat = x0 - tau0 * gradf(A(x0))
        # Now take the backwards step by finding a minimizer of g close to x
        x1 = proxg(x1hat, tau0)

        Dx = x1 - x0
        d1 = A(x1)
        f1 = f(d1)

        # Non-monotone backtracking line search, used to guarantee convergence and balance out adaptive search if
        # stepsizes grow too large
        if backtrack:
            # Find the maximum of the last `window` values of f
            M = np.max(f_hist[-min(i, window)])

            # Track the number of total backtracks
            backtrack_count = 0

            while f1 - (M + np.real(np.dot(Dx, gradf0)) + np.norm(Dx)**2 / (2 * tau0)) > EPSILON \
                    and backtrack_count < max_backtracks or not np.isreal(f1):
                # We've gone too far, so shrink the stepsize and try FBS again
                tau0 *= stepsize_shrink

                # Redo the FBS step
                x1hat = x0 - tau0 * gradf0
                x1 = proxg(x1hat, tau0)

                # Recalculate values
                d1 = A(x1)
                f1 = f(d1)
                Dx = x1 - x0

                backtrack_count += 1

            total_backtracks += backtrack_count

        # Record convergence information
        tau_hist[i] = tau0
        residual_hist[i] = np.norm(Dx) / tau0

        normalizer = np.max(np.norm(gradf0), np.norm(x1 - x1hat) / tau0) + EPSILON
        norm_residual_hist[i] = residual_hist[i] / normalizer

        f_hist[i] = f1

        max_residual = np.max(max_residual, residual_hist[i])

        if stop_rule(i, residual_hist[i], norm_residual_hist[i], max_residual, tolerance):
            break

        # Use FISTA-type acceleration, which overestimates ("predicts") the current iterate in the direction it moved
        # in the previous iteration
        if accelerate:
            # Rename last round's current variables to this round's previous variables
            x_accel0 = x_accel1
            d_accel0 = d_accel1

            x_accel1 = x1
            d_accel1 = d1

            alpha0 = alpha1

            # TODO: figure out if this should be epsilon
            # Prevent alpha from growing too large by restarting the acceleration
            if restart and (x1 - x0).transpose() * (x1 - x_accel0) > EPSILON:
                alpha0 = 1

            # Recalculate acceleration parameter
            alpha1 = (1 + np.sqrt(1 + 4*alpha0**2)) / 2

            # Overestimate the next value of x by a factor of (alpha0 - 1) / alpha
            x1 = x_accel1 + (alpha0 - 1) / alpha1 * (x_accel1 - x_accel0)
            d1 = d_accel1 + (alpha0 - 1) / alpha1 * (d_accel1 - d_accel0)

            f_hist[i] = f(d1)

        # Compute the next iteration's gradient
        gradf1 = At(gradf(d1))
        tau1 = tau0

        # Adaptive adjustments of stepsize using the Barzilai-Borwein method (spectral method), which
        # approximates the function as a quadratic, for which the ideal stepsize is simply 1/a, and dynamically
        # select a new stepsize for each iteration
        if adaptive:
            Dg = gradf1 + (x1hat - x0) / tau0
            dotprod = np.real(np.dot(Dx, Dg))

            tau_s = np.norm(Dx) ** 2 / dotprod
            tau_m = np.max(dotprod / np.norm(Dg) ** 2, 0)

            # Use an adaptive combination of tau_s and tau_m
            if 2 * tau_m > tau_s:
                tau1 = tau_m
            else:
                tau1 = tau_s - .5 * tau_m

            # Ensure non-negative stepsize
            if tau1 <= 0 or np.isinf(tau1) or np.isnan(tau1):
                tau1 = tau0 * 1.5

        i += 1

    result = {
        'residuals': residual_hist,
        'norm_residuals': norm_residual_hist,
        'stepsizes': tau_hist,
        'backtracks_count': total_backtracks,
        'iteration_count': i,
    }

    return type('FASTAResult', (object,), result)
