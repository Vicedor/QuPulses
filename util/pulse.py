"""
Implements a quantum pulse with pulse shape u(t) and coupling constant from cavity g(t)
"""
import numpy as np
from HelperFunctionality.constants import *
import HelperFunctionality.math_functions as m


class Pulse:
    def __init__(self, shape: str, in_going: bool, args):
        self.shape = shape
        self.in_going: bool = in_going
        self.args = args
        if shape == gaussian:
            self._u = m.gaussian(*args)
            self._g = m.gaussian_integral(*args)
        elif shape == gaussian_sine:
            self._u = m.gaussian_sine(*args)
            self._g = m.gaussian_sine_integral(*args)
        elif shape == filtered_gaussian:
            self._u = m.filtered_gaussian(*args)
            self._g = m.filtered_gaussian_integral(*args)

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
