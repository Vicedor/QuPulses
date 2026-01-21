"""
opo.py

Implements the operator transform for a one-side Optical Parametric Oscillator (OPO) of the form:

b_out(w) = int dw' F(w, w') a_in(w') + int G^*(w, w') a_in^dag(w')

The form for F and G can be found by following the derivation in [1] chapter 10.2. The F and G functions are used
to find the mode transformation of an input mode u and up to two output modes ([v] or [v1, v2]). This functionality
is used in conjunction with the methods used in [2].

[1] Gardiner, C.W, and Crispin W Gardiner. Quantum Noise. Springer, 1991.
[2] {insert reference for published paper}
"""

import numpy as np
from helper_functions import overlap

from typing import Callable, Tuple, List, Union


class OpticalParametricOscillator:
    """
    Implements the transformation of input pulse and output pulse(s) for a one-sided Optical Parametric Oscillator
    (OPO), following the derivation in [1], and computes the folding of the input and output pulses with F and G as
    defined in [2].

    This class stores the parameters for the OPO and the transformation functions F and G. Since the F and G functions
    for an OPO have a delta function in w', the integrals are easily computed analytically, and therefore the
    transformed functions fu, gu etc. are computed analytically except for the norm given by zeta_u, xi_u, which is
    computed numerically.

    Parameters
    ----------
    gamma : float
        The decay rate of the one-sided cavity.
    xi : complex
        The squeezing strength of the non-linear parametric amplifier in the cavity (xi = r * e^(i * phi)).
    Delta : float
        The detuning of the cavity with respect to the input pulse.
    """
    def __init__(self, gamma: float, xi: complex, Delta: float):
        # Physical parameters
        self.gamma = gamma      # decay rate of the open quantum system
        self.xi = xi            # amplitude strength of the parametric oscillator
        self.Delta = Delta      # detuning between the parametric oscillators and the input pulse

        # Calculate the F and G functions as found in chapter 10.2 in [1]
        denominator = lambda w: self.Delta ** 2 + (self.gamma / 2 - 1j * w) ** 2 - self.xi * np.conjugate(self.xi)
        num1 = lambda w: (self.gamma / 2 - 1j * self.Delta) ** 2 + w**2 + self.xi * np.conjugate(self.xi)
        num2 = self.xi * self.gamma

        self.F = lambda w: - num1(w) / denominator(w)             # times delta(w - w')
        self.G = lambda w: - np.conjugate(num2 / denominator(w))  # times delta(w + w'), so evaluate all product at -w

    def get_fu_and_gu(self, omegas: np.ndarray, u: Callable[[float], complex])\
            -> Tuple[float, float,
            Callable[[float | np.ndarray], complex | np.ndarray],
            Callable[[float | np.ndarray], complex | np.ndarray]]:
        """
        Computes the overlap of u with F and G as defined in [2]:

        fu(w) = int dw' F(w, w') u(w') / zeta_u
        gu(w) = int dw' G*(w, w') u*(w') / xi_u

        Parameters
        ----------
        omegas : np.ndarray
            Array of frequencies to compute the numerical integral int dw |fu(w)|^2 and int dw |gu(w)|^2 to find
            zeta_u and xi_u.
        u : callable[[float], complex]
            The input mode function.

        Return
        ------
        tuple[float, float, callable, callable]
            zeta_u, xi_u, fu, gu in that order.
        """
        # Define fu and gu which is not normalized yet. Integral over delta function shifts arguments
        fu_temp = lambda omega: self.F(omega) * u(omega)
        gu_temp = lambda omega: np.conjugate(self.G(omega)) * np.conjugate(u(- omega))

        # Find the norm of fu and gu and redefine the normalized functions
        zeta_u = np.sqrt(overlap(fu_temp, fu_temp, omegas))
        xi_u = np.sqrt(overlap(gu_temp, gu_temp, omegas))
        fu = lambda omega: fu_temp(omega) / zeta_u
        gu = lambda omega: gu_temp(omega) / xi_u

        return zeta_u, xi_u, fu, gu

    def get_fv_and_gv(
            self,
            omegas: np.ndarray,
            v: Union[Callable[[float], complex], List[Callable[[float], complex]]]
    ):
        """
        Computes the overlap of v (v1, v2) with F and G as defined in [2]:

        fv(w) = int dw' F*(w, w') v(w') / zeta_v
        gv(w) = int dw' G(w, w') v*(w') / xi_v

        Parameters
        ----------
        omegas : np.ndarray
            Array of frequencies to compute the numerical integral int dw |fv(w)|^2 and int dw |gv(w)|^2 to find
            zeta_v and xi_v.
        v : callable[[float], complex] | list[callable[[float], complex]]
            The input mode function(s). Accepts a mode function or a list of length 1 or 2 of mode functions, for the
            case of 1 or 2 output modes.

        Return
        ------
        tuple[float, float, callable, callable] |
         tuple[float, float, float, float, callable, callable, callable, callable]
            zeta_v, xi_v, fv, gv for a single output pulse or zeta_v1, xi_v1, zeta_v2, xi_v2, fv1, gv1, fv2, gv2 for
            two output pulses. The output is given in that order.
        """
        if not isinstance(v, list):
            v = [v]
        if len(v) == 1:
            # Transform the output mode with F and G, still not normalized
            fv_temp = lambda omega: np.conjugate(self.F(omega)) * v(omega)
            gv_temp = lambda omega: self.G(omega) * np.conjugate(v(- omega))

            # Find the norm and normalize fv and gv
            zeta_v = np.sqrt(overlap(fv_temp, fv_temp, omegas))
            xi_v = np.sqrt(overlap(gv_temp, gv_temp, omegas))
            fv = lambda omega: fv_temp(omega) / zeta_v
            gv = lambda omega: gv_temp(omega) / xi_v

            return zeta_v, xi_v, fv, gv
        elif len(v) == 2:
            # Unpack the v's
            v1, v2 = v

            # Transform the output modes with F and G, still not normalized
            fv1_temp = lambda omega: np.conjugate(self.F(omega)) * v1(omega)
            gv1_temp = lambda omega: self.G(omega) * np.conjugate(v1(- omega))
            fv2_temp = lambda omega: np.conjugate(self.F(omega)) * v2(omega)
            gv2_temp = lambda omega: self.G(omega) * np.conjugate(v2(- omega))

            # Find the norms and normalize fvs and gvs
            zeta_v1 = np.sqrt(overlap(fv1_temp, fv1_temp, omegas))
            xi_v1 = np.sqrt(overlap(gv1_temp, gv1_temp, omegas))
            zeta_v2 = np.sqrt(overlap(fv2_temp, fv2_temp, omegas))
            xi_v2 = np.sqrt(overlap(gv2_temp, gv2_temp, omegas))

            fv1 = lambda omega: fv1_temp(omega) / zeta_v1
            gv1 = lambda omega: gv1_temp(omega) / xi_v1
            fv2 = lambda omega: fv2_temp(omega) / zeta_v2
            gv2 = lambda omega: gv2_temp(omega) / xi_v2

            return zeta_v1, xi_v1, zeta_v2, xi_v2, fv1, gv1, fv2, gv2
        else:
            raise ValueError('The maximum number of output modes are 2.')
