"""A Python implementation of Goldstein et. al's Fast Adaptive Shrinkage/Thresholding Algorithm for convex optimization.

FASTA minimizes a function expressed in the form,

    min h(x) = f(Ax) + g(x),

where `f` is convex and differentiable and `g` is convex.
`g` may be non-differentiable, or possibly not even finite-valued, so normal gradient descent methods cannot be used.

The FASTA algorithm is a variant of the Forward-Backward Splitting (FBS) algorithm, which alternatively takes steps
to optimize `h` by optimizing `f` and then `g`. The optimization of `g` relies on the computation of the proximal
operator of `g` with stepsize `t`, equal to argmin { t*g(x) + .5 ||x-z||**2 }, which finds a point `x` close to the
minimum value of `g` while not straying too far from `z`.

The FASTA algorithm incorporates various improvements to FBS, as detailed in (Goldstein et al. 2016), including
adaptive stepsize selection, acceleration, non-monotone backtracking line search,
and a variety of stopping conditions. Additionally, included in this pacakage are eleven different example problems that
are solved using FASTA.
"""

from typing import Callable
import numpy as np
import numpy.linalg as la

import flow

__author__ = "Noah Singer"

__all__ = ["fasta", "Convergence"]

def fasta(A,
          f: Callable[[np.ndarray], float], gradf: Callable[[np.ndarray], np.ndarray],
          g: Callable[[np.ndarray], float], proxg: Callable[[np.ndarray], np.ndarray], x0: np.ndarray,

          adaptive: bool=True, accelerate: bool=False, verbose: bool=True,
          max_iters: int=1000, tolerance: float=1e-5,

          backtrack: bool=True, stepsize_shrink: bool=None, window: int=10, max_backtracks: int=20,
          restart: bool=True,

          evaluate_objective: bool=False, record_iterates: bool=False,
          func: Callable[[np.ndarray], np.ndarray]=None):
    # Global flow variables
    x = flow.Var('x', "The iterate.")
    z = flow.Var('z', "The conditioned iterate.")
    gradfx = flow.Var('gradfx', "The value of gradf(x).")
    tau = flow.Var('tau', "The stepsize.")
    residual = flow.Var('residual', "The residual.")
    norm_residual = flow.Var('norm_residual', "The normalized residuals.")
    objective = flow.Var('objective', "The objective.")

    xhat = flow.Var('xhat', "The partial iterate after the forward step.")
    fx = flow.Var('f', "The value of f(x).")
    Dx = flow.Var('Dx', "The different between the current and last iterates.")

    @flow.flow
    def stepsize_estimator(inp, state):
        """A flow to estimate the step size for gradient descent.
        Computes the norm of the difference in the gradient evaluated at two random points in order to approximate the Lipschitz
        constant of f."""
        # Compute two random vectors
        x1 = np.random.randn(*state[x].shape)
        x2 = np.random.randn(*state[x].shape)

        # Compute the gradients at the two vectors
        gradf1 = inp['A'].H(inp['gradf'](inp['A'](x1)))
        gradf2 = inp['A'].H(inp['gradf'](inp['A'](x2)))

        # Approximate the Lipschitz constant of gradf
        L = la.norm((gradf1 - gradf2).ravel()) / la.norm((x1 - x2).ravel())

        # We're guaranteed that FBS converges for tau <= 2.0 / L
        state[tau] = (2 / L) / 10

    @flow.flow
    def initializer(inp, state):
        """A flow to initialize the loop variables for FASTA."""
        state[z] = inp['A'](state[x])
        state[fx] = inp['f'](state[z])
        state[gradfx] = inp['A'].H(inp['gradf'](state[z]))

    @flow.flow
    def fbs(inp, state):
        """A flow to perform forwards/backwards splitting (FBS)."""
        # Forward step (gradient step)
        state[xhat] = state[fasta_loop, x, -1] - state[gradfx] * state[tau]
        # Backward step (proximal step)
        state[x] = inp['proxg'](state[xhat], state[tau])

        # Update loop variables
        state[Dx] = state[x] - state[fasta_loop, x, -1]
        state[z] = inp['A'](state[x])
        state[fx] = inp['f'](state[z])
        state[gradfx] = inp['A'].H(inp['gradf'](state[z]))

    @flow.flow
    def convergence(inp, state):
        """A flow to record algorithm convergence information."""
        state[residual] = la.norm(state[Dx].ravel()) / state[tau]
        normalizer = max(la.norm(state[gradfx, -1].ravel()), la.norm((state[x] - state[xhat]).ravel()) / state[tau, -1])\
                     + inp['epsilon']
        state[norm_residual] = state[residual] / normalizer
        state[objective] = state[fx] + inp['g'](state[x])

    @flow.flow
    def residual_stop(inp, state):
        """A flow to check whether the iteration should continue."""
        state[flow.CONDITION] = state[residual] >= inp['tolerance'] and state[i] <= inp['max_iterations']

    @flow.flow
    def adaptive_stepsize_selector(inp, state):
        """A flow to heuristically select a stepsize by approximating the Hessian as the identity."""
        Dg = state[gradfx] + (state[xhat] - state[x,-1]) / state[tau]
        dotprod = np.real(state[Dx].ravel().T @ Dg.ravel())

        # One least squares estimate of the best stepsize
        tau_s = la.norm(state[Dx].ravel()) ** 2 / dotprod
        # A different least squares estimate of the best stepsize
        tau_m = max(dotprod / la.norm(Dg.ravel()) ** 2, 0)

        # Use an adaptive combination of tau_s and tau_m
        if 2 * tau_m > tau_s:
            state[tau] = tau_m
        else:
            state[tau] = tau_s - .5 * tau_m

        # Ensure non-negative stepsize
        if state[tau] <= 0 or np.isinf(state[tau]) or np.isnan(state[tau]):
            state[tau] = state[tau,-1] * 1.5

    # Backtracking
    M = flow.Var('M', "The maximum value of f over the last several iterations.")

    @flow.flow
    def backtracking_condition(inputs, state):
        state[flow.CONDITION] = state[fx] - (state[M] + np.real(state[Dx].ravel().T @ state[-2, gradfx, -1].ravel()) + la.norm(state[Dx].ravel()) ** 2 / (2 * state[-2, tau, -1])) > inputs['epsilon'] and state[state.loops[-1].counter] <= inputs['max_backtracks']

    @flow.flow
    def backtracking_window_estimator(inputs, state):
        state[M] = np.max(state[fx, max(state[i] - inputs['window'] + 1, 0):(state[i] + 1)])

    @flow.flow
    def backtracking_stepsize_updater(inputs, state):
        state[tau] *= inputs['stepsize_shrink']

    backtracker = backtracking_window_estimator >> flow.Loop(backtracking_stepsize_updater >> fbs, backtracking_condition)

    # Acceleration
    x_unaccel = flow.Var('x_unaccelerated', "The un-accelerated iterate.")
    z_unaccel = flow.Var('z_unaccelerated', "The un-accelerated conditioned iterate.")
    alpha = flow.Var('alpha', "The acceleration parameter.")
    accel_vars = [x_unaccel, z_unaccel, alpha]

    @flow.flow
    def acceleration_initializer(inp, state):
        """A flow to initialize the acceleration variables."""
        state[x_unaccel] = state[x]
        state[z_unaccel] = state[z]
        state[alpha] = 1.0

    @flow.flow
    def accelerator(inp, state):
        """A flow to accelerate the convergence of FBS by overestimating the iterates."""
        # Remember the un-accelerated variables
        state[x_unaccel] = state[x]
        state[z_unaccel] = state[z]
        alpha0 = state[alpha]

        # Prevent alpha from growing too large by restarting the acceleration
        if inp['restart'] and (state[x,-1] - state[x]).ravel().T @ (state[x] - state[x_unaccel, -1]).ravel() > 1E-30:
            alpha0 = 1.0

        # Recalculate acceleration parameter
        state[alpha] = (1 + np.sqrt(1 + 4 * alpha0 ** 2)) / 2

        # Overestimate the next value of x by a factor of (alpha0 - 1) / alpha
        # NOTE: this makes a copy of x1, which is necessary since x1's reference is linked to x0
        state[x] = state[x] + (alpha0 - 1) / state[alpha] * (state[x_unaccel] - state[x_unaccel, -1])
        state[z] = state[z] + (alpha0 - 1) / state[alpha] * (state[z_unaccel] - state[z_unaccel, -1])

        state[fx] = inp['f'](state[z])

    if stepsize_shrink is None and backtrack:
        if adaptive:
            # This is more aggressive, since the stepsize increases dynamically
            stepsize_shrink = 0.2
        else:
            stepsize_shrink = 0.5

    body = fbs >>\
           (backtracker if backtrack else None) >>\
           (adaptive_stepsize_selector if adaptive else None) >>\
           (accelerator if accelerate else None)\
           >> convergence

    loop_vars = [x, fx, gradfx, tau, flow.TIME, residual, norm_residual, objective]
    initial_vars = [x, z, fx, gradfx, tau]
    if accelerate:
        loop_vars += accel_vars
        initial_vars += accel_vars

    fasta_loop = flow.Loop(flow.time(body), residual_stop, loop_vars, save=True, check_first=False, initial_vars=initial_vars)
    fasta_flow = stepsize_estimator >> initializer >> (acceleration_initializer if accelerate else None) >> fasta_loop

    i = fasta_loop.counter

    fasta_state = flow.State()
    fasta_state[x] = x0

    fasta_flow.operate({
        'A': A,
        'f': f,
        'gradf': gradf,
        'g': g,
        'proxg': proxg,
        'max_iterations': max_iters,
        'tolerance': tolerance,
        'epsilon': 1e-8,
        'restart': restart,
        'stepsize_shrink': stepsize_shrink,
        'max_backtracks': max_backtracks,
        'window': window
    }, fasta_state)

    return type('Convergence', (object,), {
        'residuals': fasta_state.tapes[fasta_loop][residual,:],
        'norm_residuals': fasta_state.tapes[fasta_loop][norm_residual,:],
        'times': fasta_state.tapes[fasta_loop][flow.TIME,:],
        'objectives': fasta_state.tapes[fasta_loop][objective,:],
        'iteration_count': fasta_state[i],
        'iterates': fasta_state.tapes[fasta_loop][x,:],
        'solution': fasta_state[x]
    })
