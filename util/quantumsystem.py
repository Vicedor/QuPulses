"""
This file contains an abstract class to be defined for each kind of interferometer. The abstract interferometer class
will contain functions for defining the Hilbert space and the SLH components
"""
import qutip as qt
import numpy as np
from abc import ABCMeta, abstractmethod
import SLH.network as nw
import util.pulse as p
from typing import List, Any, Callable, Union


class QuantumSystem(metaclass=ABCMeta):
    def __init__(self, psi0: qt.Qobj, pulses: List[p.Pulse], times: np.ndarray,
                 options: qt.Options = qt.Options(nsteps=1000000000, store_states=1, atol=1e-8, rtol=1e-6)):
        """
        Initializes the Quantum System object with a specific initial state, pulse shape and array of times
        :param psi0: The initial state for the system
        :param pulses: The pulses used in the quantum system
        :param times: An array of times at which to evaluate the expectation values and states of the system
        :param options: Options for the integrator as a qutip Options object
        """
        self._psi0: qt.Qobj = psi0
        self._pulses: List[p.Pulse] = pulses
        self._times: np.ndarray = times
        self._options: qt.Options = options

    @property
    def pulses(self):
        return self._pulses

    @pulses.setter
    def pulses(self, value):
        self._pulses = value

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, value):
        self._times = value

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, value):
        self._options = value

    def redefine_pulse_args(self, args):
        """
        Redefines all pulses with new arguments. Sets all pulses with the same arguments
        :param args: The arguments to give to the pulses
        """
        for pulse in self._pulses:
            pulse.set_pulse_args(args)

    @property
    def psi0(self):
        return self._psi0

    @psi0.setter
    def psi0(self, value):
        self._psi0 = value

    @abstractmethod
    def create_component(self) -> nw.Component:
        """
        Creates the SLH component for the interferometer using the SLH composition rules implemented in the SLH package
        :return: The total SLH-component for the interferometer
        """
        pass

    @abstractmethod
    def get_expectation_observables(self) -> Union[List[qt.Qobj], Callable]:
        """
        Gets a list of the observables to evaluate the expectation value of at different times in the time-evolution
        :return: The list of observables to get the expectation values of
        """
        pass

    @abstractmethod
    def get_plotting_options(self) -> Any:
        """
        Gets the plotting options for the pulses and excitation-content of the system
        :return: The plotting options
        """
        pass
