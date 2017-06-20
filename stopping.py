"""A collection of various stopping conditions for the FASTA algorithm.

Each condition will be passed the iteration number, the residual, the normalized residual, the maximum residual,
and the tolerance."""


def residual(i, resid, norm_resid, max_resid, tolerance):
    """Stop when the residual becomes small."""
    return resid < tolerance


def norm_residual(i, resid, norm_resid, max_resid, tolerance):
    """Stop when the normalized residual becomes small."""
    return norm_resid < tolerance


def ratio_residual(i, resid, norm_resid, max_resid, tolerance):
    """Stop when the ratio of the current residual and maximum residual seen becomes small."""
    return resid / max_resid < tolerance


def hybrid_residual(i, resid, norm_resid, max_resid, tolerance):
    """Stop when either the normalized residual or the ratio of current and maximum residuals becomes small."""
    return resid / max_resid < tolerance or norm_resid < tolerance
