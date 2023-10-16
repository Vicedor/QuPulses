"""
Implements a quantum pulse with pulse shape u(t) and coupling constant from cavity g(t)
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from util.constants import *
import util.math_functions as m
from typing import Tuple, Callable, List


class Pulse:
    def __init__(self, shape: str, in_going: bool, args):
        self.shape = shape
        self.in_going: bool = in_going
        self._u, self._g = self._get_mode_function(args)

    def set_pulse_args(self, args):
        """
        Redefines the pulse-mode with new arguments. Does not change the functional shape of the pulse, only the
        arguments for the mode function
        :param args: The new arguments
        """
        self._u, self._g = self._get_mode_function(args)

    def _get_mode_function(self, args) -> Tuple[Callable[[float], float], Callable[[float], float]]:
        """
        Gets the mode functions u and g from the shape attribute
        :return: The mode functions u and g
        """
        if self.shape == gaussian:
            u = m.gaussian(*args)
            g = m.gaussian_squared_integral(*args)
        elif self.shape == gaussian_sine:
            u = m.gaussian_sine(*args)
            g = m.gaussian_sine_integral(*args)
        elif self.shape == filtered_gaussian:
            u = m.filtered_gaussian(*args)
            g = m.filtered_gaussian_integral(*args)
        elif self.shape == n_filtered_gaussian:
            u = m.n_filtered_gaussian(*args)
            g = m.n_filtered_gaussian_integral(*args)
        elif self.shape == exponential:
            u = m.exponential(*args)
            g = m.exponential_integral(*args)
        elif self.shape == hermite_gaussian:
            gauss = m.gaussian(args[0], args[1])
            hermite = m.normalized_hermite_polynomial(*args)
            u = lambda t: gauss(t) * hermite(t)
            g = m.hermite_gaussian_integral(*args)
        elif self.shape == frequency_mod_gaussian:
            u = m.freq_mod_gaussian(*args)
            g = m.gaussian_squared_integral(args[0], args[1])
        elif self.shape == two_modes_gaussian:
            u = m.gaussian(*args)
            g = m.two_mode_integral(*args)
        elif self.shape == two_modes_sine:
            u = m.gaussian_sine(*args)
            g = m.two_mode_integral(*args)
        elif self.shape == undefined:
            u = args[0]
            g = args[1]
        else:
            raise ValueError(self.shape + " is not a defined pulse mode.")
        return u, g

    def u(self, t: float) -> float:
        """
        Evaluates the pulse shape u(t) depending on the pulse shape specified
        :param t: The time to evaluate the pulse shape at
        :return: The normalized pulse shape value at the specified time
        """
        return self._u(t)

    def g(self, t: float) -> float:
        """
        Evaluates the g(t) function (eq. 2 in the InteractionPicturePulses paper) given the specified pulse shape
        :param t: The time at which the function is evaluated
        :return: The value of g_u(t) at the specified time
        """
        real = False
        temp = np.conjugate(self.u(t))
        if np.imag(temp) == 0:
            real = True
        # Divide by the integral of u(t) only if u(t) is not very small (to avoid numerical instability)
        if abs(temp) >= epsilon:
            if self.in_going:
                temp /= np.sqrt(10e-6 + 1 - self._g(t))
            else:
                temp = - temp / np.sqrt(10e-6 + self._g(t))
        # If t=0 the out_going g(t) function is infinite, so set it to 0 to avoid numerical instability
        if not self.in_going and abs(temp) >= 1000000:
            temp = 0
        if real:
            temp = np.real(temp)
        return temp

    def split_g(self, dt, split_between, index) -> Callable[[float], float]:
        """
        Splits the g into distinct function, where the pulses take turns of having non-zero g to make their overlap
        always 0.
        :param dt: The time interval each function gets to be non-zero
        :param split_between: The number of pulses to split the time interval between
        :param index: The index of when it is this pulse's turn to be non-zero
        :return: A function obeying the splitting of the time
        """
        def f(t):
            if (t // dt) % split_between == index:
                return self.g(t)
            else:
                return 0
        return f


def _transform_input_pulse_across_cavity(u_target: np.ndarray, u_cavity: np.ndarray, t_list: np.ndarray) -> np.ndarray:
    """
    Does the single transformation across one input cavity (equation A14 in Fan's paper).
    :param u_target: The target pulse shape after the transformation across the cavity (phi_m^(n))
    :param u_cavity: The pulse shape emitted by the cavity to transform across (phi_n^(n))
    :param t_list: The list of times for which the pulses are defined
    :return: The pulse shape to emit to become the target pulse shape after travelling past the cavity (phi_m^(n-1))
    """
    mode_overlap = cumulative_trapezoid(u_cavity.conjugate() * u_target, t_list, initial=0)
    u_cavity_int = cumulative_trapezoid(u_cavity.conjugate() * u_cavity, t_list, initial=0)
    return u_target + u_cavity * mode_overlap / (1 + epsilon - u_cavity_int)


def transform_input_pulses_across_cavities(u_targets: List[np.ndarray], t_list: np.ndarray):
    """
    Transforms the target pulse shapes aimed at hitting a given system, into the pulse shapes that must be emitted
    by the M input virtual cavity in series, to be correctly scattered to the target pulse shape by the cavities
    in front
    :param u_targets: The target mode functions to hit the system. First entry is for cavity just before system.
    Last entry is for cavity furthest from system (so the mode that needs most transformation)
    :param t_list: The list of times at which the pulse modes are defined.
    :return: The actual modes to be emitted by the input virtual cavities such that they are transformed into the
     correct target modes
    """
    output_modes = []
    for i, u in enumerate(u_targets):
        u_transform = u
        for j in range(i):
            u_transform = _transform_input_pulse_across_cavity(u_transform, output_modes[j], t_list)
        output_modes.append(u_transform)
    return output_modes


def __transform_output_pulse_across_cavity(u_target: np.ndarray, u_cavity: np.ndarray, t_list: np.ndarray) -> np.ndarray:
    """
    Does the single transformation across one output cavity (equation A12 in Fan's paper).
    :param u_target: The target pulse shape after the transformation across the cavity (psi_m^(n-1))
    :param u_cavity: The pulse shape emitted by the cavity to transform across (psi_n^(n-1))
    :param t_list: The list of times for which the pulses are defined
    :return: The pulse shape to emit to become the target pulse shape after travelling past the cavity (phi_m^(n))
    """
    mode_overlap = cumulative_trapezoid(u_cavity.conjugate() * u_target, t_list, initial=0)
    u_cavity_int = cumulative_trapezoid(u_cavity.conjugate() * u_cavity, t_list, initial=0)
    return u_target - u_cavity * mode_overlap / (epsilon + u_cavity_int)


def transform_output_pulses_across_cavities(v_targets: List[np.ndarray], t_list: np.ndarray):
    """
    Transforms the target pulse shapes emitted by a given system, into the pulse shapes that must be absorbed
    by the N output virtual cavity in series, to be correctly scattered to the target pulse shape by the cavities
    in front
    :param v_targets: The target mode functions emitted from the system. First entry is for cavity just after system.
    Last entry is for cavity furthest from system (so the mode that needs most transformation)
    :param t_list: The list of times at which the pulse modes are defined.
    :return: The actual modes to be absorbed by the output virtual cavities such that they are transformed into the
     correct target modes
    """
    output_modes = []
    for i, u in enumerate(v_targets):
        u_transform = u
        for j in range(i):
            u_transform = __transform_output_pulse_across_cavity(u_transform, output_modes[j], t_list)
        output_modes.append(u_transform)
    return output_modes

