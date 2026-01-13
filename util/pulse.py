"""
Implements a quantum pulse with pulse shape u(t) and coupling constant from cavity g(t)
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
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
        elif self.shape == reverse_exponential:
            u = m.reverse_exponential(*args)
            g = m.reverse_exponential_integral(*args)
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
        if self.in_going:
            temp /= np.sqrt(epsilon + 1 - self._g(t))
        else:
            temp = - temp / np.sqrt(epsilon + self._g(t))
        # If t=0 the out_going g(t) function is infinite, so set it to 0 to avoid numerical instability
        if not self.in_going and abs(temp) >= 1000000:
            temp = 0
        if real:
            temp = np.real(temp)
        return temp


def __transform_input_pulse_across_cavity_deprecated(u_target: np.ndarray, u_cavity: np.ndarray, t_list: np.ndarray) -> np.ndarray:
    """
    Does the single transformation across one input cavity (equation A14 in Fan's paper).
    :param u_target: The target pulse shape after the transformation across the cavity (phi_m^(n))
    :param u_cavity: The pulse shape emitted by the cavity to transform across (phi_n^(n))
    :param t_list: The list of times for which the pulses are defined
    :return: The pulse shape to emit to become the target pulse shape after travelling past the cavity (phi_m^(n-1))
    """
    mode_overlap = cumulative_trapezoid(u_cavity.conjugate() * u_target, t_list, initial=0)# - trapezoid(u_cavity.conjugate() * u_target, t_list)
    u_cavity_int = cumulative_trapezoid(u_cavity.conjugate() * u_cavity, t_list, initial=0)
    return u_target + u_cavity * mode_overlap / (1 + epsilon - u_cavity_int)


def transform_input_pulses_across_cavities_deprecated(u_targets: List[np.ndarray], t_list: np.ndarray):
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
            u_transform = __transform_input_pulse_across_cavity_deprecated(u_transform, output_modes[j], t_list)
        output_modes.append(u_transform)
    return output_modes


def __calculate_first_integrals(modes: List[np.ndarray], t_list: np.ndarray):
    """
    Calculates all the first integrals between the modes, which are the only integrals that needs to be calculated
    by hand. The other integrals in the iterative process can be derived from these integrals.
    The integrals are stored in lists of the form (where the star denotes conjugate):

    [[u1*u1,            0,     0, 0, ...],
     [u1*u2, u2*u2,     0,     0, 0, ...],
     [u1*u3, u2*u3, u3*u3,     0, 0, ...],
     [u1*u4, u2*u4, u3*u4, u4*u4, 0, ...],
     ...]

    :param modes: The modes [u1, u2, u3, u4, ...]. The mode that is mostly transformed (leftmost cavity) is mode 1
    :param t_list: The list of times at which the integrals are defined
    :return: The structure of mode integrals
    """
    integrals = [[0 for _ in range(len(modes))] for _ in range(len(modes))]
    for i, u1 in enumerate(modes):
        for j, u2 in enumerate(modes[i:]):
            mode_overlap = cumulative_trapezoid(u1.conjugate() * u2, t_list, initial=0)
            integrals[i + j][i] = mode_overlap
    return integrals


def __transform_pulses_iteration(u_n: np.ndarray, u_targets: List[np.ndarray], last_integrals, is_input: bool):
    """
    Performs a single iteration of transforming the mode functions over a single cavity
    :return:
    """
    next_integrals = [[0 for _ in range(len(u_targets))] for _ in range(len(u_targets))]
    for i in range(len(u_targets)):
        for j in range(i, len(u_targets)):
            ui_uj_int = last_integrals[j + 1][i + 1]
            ui_unp1_int = np.conjugate(last_integrals[i + 1][0])
            unp1_uj_int = last_integrals[j + 1][0]
            unp1_unp1_int = last_integrals[0][0]
            if is_input:
                mode_overlap = ui_uj_int + ui_unp1_int * unp1_uj_int / (1 - unp1_unp1_int + epsilon)
            else:
                mode_overlap = ui_uj_int - ui_unp1_int * unp1_uj_int / (unp1_unp1_int + epsilon)
            next_integrals[j][i] = mode_overlap

    u_next = []
    for i, u_m in enumerate(u_targets):
        if is_input:
            integral = last_integrals[i + 1][0] / (1 - last_integrals[0][0] + epsilon)
            u_next.append(u_m + u_n * integral)
        else:
            integral = last_integrals[i + 1][0] / (last_integrals[0][0] + epsilon)
            u_next.append(u_m - u_n * integral)

    return u_next, next_integrals


def transform_pulses(u_targets: List[np.ndarray], t_list: np.ndarray, is_input: bool) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Performs the iterative transformation of a number of pulses to cavities in series which can leak the pulses
    such that the given mode functions are emitted at the end of the series of input cavities
    :param u_targets: The target mode functions to be emitted, where the first entry is transformed the least, and the
    last entry is transformed the most
    :param t_list: The times at which the target mode functions are defined
    :return: A list of the modes and a list of the integrals over the modes
    """
    u_modes = []
    u_ints = []

    u_iters = u_targets
    integrals = __calculate_first_integrals(u_targets, t_list)
    for i in range(len(u_targets)):
        u_modes.append(u_iters[0])
        u_ints.append(integrals[0][0])
        u_iters, integrals = __transform_pulses_iteration(u_iters[0], u_iters[1:], integrals, is_input=is_input)

    return u_modes, u_ints


def __transform_output_pulse_across_cavity_deprecated(u_target: np.ndarray, u_cavity: np.ndarray, t_list: np.ndarray) -> np.ndarray:
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


def transform_output_pulses_across_cavities_deprecated(v_targets: List[np.ndarray], t_list: np.ndarray):
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
            u_transform = __transform_output_pulse_across_cavity_deprecated(u_transform, output_modes[j], t_list)
        output_modes.append(u_transform)
    return output_modes

