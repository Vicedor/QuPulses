"""
This file contains an abstract class to be defined for each kind of interferometer. The abstract interferometer class
will contain functions for defining the Hilbert space and the SLH components
"""
import qutip as qt
from abc import ABCMeta, abstractmethod
import SLH.network as nw
import util.pulse as p
from typing import List, Any


class Interferometer(metaclass=ABCMeta):
    def __init__(self, psi0: qt.Qobj, pulses: List[p.Pulse]):
        self._psi0: qt.Qobj = psi0
        self._pulses: List[p.Pulse] = pulses

    @property
    def pulses(self):
        return self._pulses

    @pulses.setter
    def pulses(self, value):
        self._pulses = value

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
    def get_expectation_observables(self) -> List[qt.Qobj]:
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
