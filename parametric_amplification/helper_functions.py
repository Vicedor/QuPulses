import numpy as np
from scipy.integrate import quad
from typing import Callable


def overlap(f: Callable[[float], complex], g: Callable[[float], complex], xs: np.ndarray):
    """
    Compute the overlap integral <f|g> over an interval.

    Parameters
    ----------
    f, g : callable
        Functions of frequency returning (complex) amplitudes.
    xs : ndarray
        x-values of interval; the integration limits are xs[0] and xs[-1].

    Returns
    -------
    complex
        The value of the integration of conj(f(x)) * g(x) over the interval.
    """
    return quad(lambda omega: np.conjugate(f(omega)) * g(omega), xs[0], xs[-1], complex_func=True)[0]