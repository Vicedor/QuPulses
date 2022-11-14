"""
This file contains a component factory, which can create many of the most used components in networks
"""
import numpy as np
import qutip as qt
import SLH.network as nw
from typing import Union, Callable


def create_cavity(I: qt.Qobj, a: qt.Qobj, g: Union[float, Callable[[float], float]], w0: float) -> nw.Component:
    """
    Creates a cavity-component with the given coupling factor, energy spacing and ladder operator
    :param I: The identity operator of the total Hilbert space
    :param a: The ladder operator acting on the cavity in the total Hilbert space
    :param g: The coupling factor between the cavity and the environment (possibly time-dependent)
    :param w0: The energy spacing of the modes in the cavity
    :return: An SLH network component of a cavity with the given parameters
    """
    if isinstance(g, float):
        a_t: qt.Qobj = g * a
    else:
        a_t: qt.QobjEvo = qt.QobjEvo([[a, lambda t, args: np.conjugate(g(t))]])
    return nw.Component(nw.MatrixOperator(I), nw.MatrixOperator(a_t), w0 * a.dag() * a)


def create_beam_splitter() -> nw.Component:
    """
    Creates a beam splitter as in SLH paper
    :return: The beam splitter component
    """
    return nw.Component(S=nw.MatrixOperator([[1 / np.sqrt(2), 1 / np.sqrt(2)],
                                             [-1 / np.sqrt(2), 1 / np.sqrt(2)]]),
                        L=nw.MatrixOperator([[0], [0]]),
                        H=0)


def create_three_level_atom(I: qt.Qobj, destroy1: qt.Qobj, destroy2: qt.Qobj,
                            gamma1: float, gamma2: float, w1: float, w2: float) -> nw.Component:
    """
    Creates a three level atom in the Vee configuration (but it can also be used for other configurations)
    :param I: The identity operator of the total Hilbert space
    :param destroy1: The sigma minus operator of the total Hilbert space, destroying a quantum in left Vee branch
                     and creating a quantum in the ground state
    :param destroy2: The sigma minus operator of the total Hilbert space, destroying a quantum in right Vee branch
                     and creating a quantum in the ground state
    :param gamma1: The decay rate of the left Vee branch
    :param gamma2: The decay rate of the right Vee branch
    :param w1: The energy of the left Vee branch (above the ground state)
    :param w2: The energy of the right Vee branch (above the ground state)
    :return: An SLH component of a three level atom with the given parameters
    """
    return nw.Component(S=nw.MatrixOperator(I),
                        L=nw.MatrixOperator(np.sqrt(gamma1) * destroy1 + np.sqrt(gamma2) * destroy2),
                        H=w1 * destroy1.dag() * destroy1 + w2 * destroy2.dag() * destroy2)

