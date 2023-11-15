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
    if isinstance(g, float) or isinstance(g, int):
        a_t: qt.Qobj = g * a
    else:
        a_t: qt.QobjEvo = qt.QobjEvo([[a, lambda t, args: np.conjugate(g(t))]])
    return nw.Component(nw.MatrixOperator(I), nw.MatrixOperator(a_t), w0 * a.dag() * a)


def create_beam_splitter() -> nw.Component:
    """
    Creates a beam splitter as in SLH paper. In the SLH formalism, the beam splitter matrix is such that the reflected
    input mode corresponds to the same output mode, which is different from the convention must use, see for instance
    eq A2 in the SLH-paper and eq 6.8 in Gerry & Knight, Introductory Quantum Optics
    :return: The beam splitter component
    """
    return nw.Component(S=nw.MatrixOperator([[-1 / np.sqrt(2), 1 / np.sqrt(2)],
                                             [1 / np.sqrt(2), 1 / np.sqrt(2)]]),
                        L=nw.MatrixOperator([[0], [0]]),
                        H=0)


def create_phase_shifter(phi: float) -> nw.Component:
    """
    Creates a phase shifter as in the SLH paper, with phase shift e^i*phi
    :param phi: The angle of the phase shifter from 0 to 2*pi
    :return: The phase shifter component
    """
    return nw.Component(S=nw.MatrixOperator(np.exp(1j*phi)), L=nw.MatrixOperator(0), H=0)


def create_three_level_atom(I: qt.Qobj, sigma_ae: qt.Qobj, sigma_be: qt.Qobj,
                            gamma1: float, gamma2: float, w1: float, w2: float) -> nw.Component:
    """
    Creates a three level atom in the Lambda configuration (but it can also be used for other configurations)
    :param I: The identity operator of the total Hilbert space
    :param sigma_ae: The sigma plus operator of the total Hilbert space, destroying a quantum in the excited state |e>
                     and creating a quantum in the left state |a>: sigma_ae = |a><e|
    :param sigma_be: The sigma minus operator of the total Hilbert space, destroying a quantum in upper branch
                     and creating a quantum in the right state |b>: sigma_be = |b><e|
    :param gamma1: The decay rate to the left Lambda branch |a>
    :param gamma2: The decay rate to the right Lambda branch |b>
    :param w1: The energy of the excited state (above the ground state a with 0 energy)
    :param w2: The energy of the right Lambda branch |b> (above the ground state |a>)
    :return: An SLH component of a three level atom with the given parameters
    """
    return nw.Component(S=nw.MatrixOperator(I),
                        L=nw.MatrixOperator(np.sqrt(2)**0 * np.sqrt(gamma1) * sigma_ae + np.sqrt(gamma2) * sigma_be),
                        H=w1 * sigma_ae.dag() * sigma_ae + w2 * sigma_be * sigma_be.dag())


def create_interferometer_with_lower_system(system: nw.Component) -> nw.Component:
    """
    Creates an interferometer with the given system placed in the lower arm
    :param system: The system to place in the lower arm
    :return: An interferometer component with the system in the lower arm
    """
    beam_splitter: nw.Component = create_beam_splitter()
    padded_system: nw.Component = nw.padding_top(1, system)
    return nw.series_product(nw.series_product(beam_splitter, padded_system), beam_splitter)


def create_squeezing_cavity(I: qt.Qobj, a: qt.Qobj, gamma: float, Delta: float, xi: float) -> nw.Component:
    return nw.Component(S=nw.MatrixOperator(I), L=nw.MatrixOperator(gamma * a),
                        H=Delta * a.dag() * a + 0.5j * xi * (a.dag() * a.dag() - a * a))
