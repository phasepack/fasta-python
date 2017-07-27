"""A collection of various stopping conditions for the FASTA algorithm."""

__author__ = "Noah Singer"


def residual(i: int, resid: float, norm_resid: float, max_resid: float, tolerance: float):
    """Stop when the residual becomes small.

    :param i: The current iteration number
    :param resid: The current residual
    :param norm_resid: The current normalized residual
    :param max_resid: The largest residual seen so far
    :param tolerance: The stopping tolerance
    """
    return resid < tolerance


def norm_residual(i: int, resid: float, norm_resid: float, max_resid: float, tolerance: float):
    """Stop when the normalized residual becomes small.

    :param i: The current iteration number
    :param resid: The current residual
    :param norm_resid: The current normalized residual
    :param max_resid: The largest residual seen so far
    :param tolerance: The stopping tolerance
    """
    return norm_resid < tolerance


def ratio_residual(i: int, resid: float, norm_resid: float, max_resid: float, tolerance: float):
    """Stop when the ratio of the current residual and maximum residual seen becomes small.

    :param i: The current iteration number
    :param resid: The current residual
    :param norm_resid: The current normalized residual
    :param max_resid: The largest residual seen so far
    :param tolerance: The stopping tolerance
    """
    return resid / max_resid < tolerance


def hybrid_residual(i: int, resid: float, norm_resid: float, max_resid: float, tolerance: float):
    """Stop when either the normalized residual or the ratio of current and maximum residuals becomes small.

    :param i: The current iteration number
    :param resid: The current residual
    :param norm_resid: The current normalized residual
    :param max_resid: The largest residual seen so far
    :param tolerance: The stopping tolerance
    """
    return resid / max_resid < tolerance or norm_resid < tolerance
