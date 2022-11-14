"""
This file contains an abstract class to be defined for each kind of interferometer. The abstract interferometer class
will contain functions for defining the Hilbert space and the SLH components
"""
import qutip as qt
from abc import ABCMeta, abstractmethod
import SLH.network as nw
from typing import List, Any


class Interferometer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self._psi0 = None
        pass

    @property
    @abstractmethod
    def pulses(self):
        pass

    @property
    @abstractmethod
    def psi0(self):
        pass

    @abstractmethod
    def set_pulses(self, tp: float, tau: float):
        """
        Redefines the pulses used in the interferometer
        :param tp: The time at which the gaussian pulse peaks
        :param tau: The width of the gaussian pulse
        """
        pass

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

    @psi0.setter
    def psi0(self, value):
        self._psi0 = value

    @psi0.getter
    def psi0(self):
        return self._psi0
