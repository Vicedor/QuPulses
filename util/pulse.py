"""
Implements a quantum pulse with pulse shape u(t) and coupling constant from cavity g(t)
"""
import numpy as np
from util.constants import *
import util.math_functions as m
from typing import Tuple, Callable


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
            g = m.gaussian_integral(*args)
        elif self.shape == gaussian_sine:
            u = m.gaussian_sine(*args)
            g = m.gaussian_sine_integral(*args)
        elif self.shape == filtered_gaussian:
            u = m.filtered_gaussian(*args)
            g = m.filtered_gaussian_integral(*args)
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
        temp = np.conjugate(self.u(t))
        # Divide by the integral of u(t) only if u(t) is not very small (to avoid numerical instability)
        if abs(temp) >= epsilon:
            if self.in_going:
                temp /= np.sqrt(1 - self._g(t))
            else:
                temp = - temp / np.sqrt(self._g(t))
        # If t=0 the out_going g(t) function is infinite, so set it to 0 to avoid numerical instability
        if not self.in_going and t == 0:
            temp = 0
        return temp
