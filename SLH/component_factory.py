"""
This file contains a component factory, which can create many of the most used components in networks
"""
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import SLH.network as nw
from typing import Union, Callable, List


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


def create_two_sided_cavity(I: qt.Qobj, a1: qt.Qobj, a2: qt.Qobj, g1: Union[float, Callable[[float], float]],
                            g2: Union[float, Callable[[float], float]], w01: float, w02: float) -> nw.Component:
    """
    Creates a two-sided cavity component with the given coupling factors, energy spacing and ladder operator
    :param I: The identity operator of the total Hilbert space
    :param a: The ladder operator acting on the cavity in the total Hilbert space
    :param g1: The first coupling factor between the cavity and the environment (possibly time-dependent)
    :param g2: The second coupling factor between the cavity and the environment (possibly time-dependent)
    :param w0: The energy spacing of the modes in the cavity
    :return: An SLH network component of a cavity with the given parameters
    """
    if isinstance(g1, float) or isinstance(g1, int):
        a_t1: qt.Qobj = g1 * a1
    else:
        a_t1: qt.QobjEvo = qt.QobjEvo([[a1, lambda t, args: np.conjugate(g1(t))]])
    if isinstance(g2, float) or isinstance(g2, int):
        a_t2: qt.Qobj = g2 * a2
    else:
        a_t2: qt.QobjEvo = qt.QobjEvo([[a2, lambda t, args: np.conjugate(g2(t))]])
    return nw.Component(nw.MatrixOperator([[I, 0*I], [0*I, I]]), nw.MatrixOperator([[a_t1], [a_t2]]), w01 * a1.dag() * a1 + w02 * a2.dag() * a2)


def create_beam_splitter(S: List[List[complex]] = None) -> nw.Component:
    """
    Creates a beam splitter as in SLH paper. In the SLH formalism, the beam splitter matrix is such that the reflected
    input mode corresponds to the same output mode, which is different from the convention must use, see for instance
    eq A2 in the SLH-paper and eq 6.8 in Gerry & Knight, Introductory Quantum Optics
    :return: The beam splitter component
    """
    if S is None:
        S: List[List[complex]] = [[1 / np.sqrt(2), -1 / np.sqrt(2)],
                                  [1 / np.sqrt(2), 1 / np.sqrt(2)]]
    return nw.Component(S=nw.MatrixOperator(S),
                        L=nw.MatrixOperator([[0] for _ in range(len(S))]),
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
    """
    Creates a Parametric Amplifier with Hamiltonian H = Delta a^dag * a + i*xi / 2 * (a^dag^2 - a^2) coupled to
    external degrees of freedom with rate gamma.
    :param I: The Identity operator on the full Hilbert space
    :param a: The annihilation operator of the cavity in the full Hilbert space
    :param gamma: The decay rate of the cavity
    :param Delta: The detuning of the cavity
    :param xi: The strength of the non-linear term
    :return: An SLH component for a Parametric Amplifier
    """
    return nw.Component(S=nw.MatrixOperator(I), L=nw.MatrixOperator(np.sqrt(gamma) * a),
                        H=Delta * a.dag() * a + 0.5j * xi * (a.dag() * a.dag() - a * a))


def atom_in_cavity(I: qt.Qobj, a: qt.Qobj, c: qt.Qobj, gamma: float, Delta: float,
                   Omega: float, g: float) -> nw.Component:
    """
    Creates an atom in a one-sided cavity with Hamiltonian H = Delta a^dag * a + Omega / 2 * sigma_z
    + g * (c * a^dag + c^dag * a) coupled to external degrees of freedom with rate gamma.
    :param I: The Identity operator on the full Hilbert space
    :param a: The annihilation operator of the cavity in the full Hilbert space
    :param c: The sigma minus operator for the atom
    :param gamma: The decay rate of the cavity
    :param Delta: The detuning of the cavity
    :param Omega: Atomic transition frequency
    :param g: Coupling constant between cavity and atom
    :return: An SLH component for the atom in the cavity
    """
    sigma_z: qt.Qobj = c * c.dag() - c.dag() * c
    return nw.Component(S=nw.MatrixOperator(I), L=nw.MatrixOperator(np.sqrt(gamma) * a),
                        H=Delta * a.dag() * a + Omega / 2 * sigma_z + 1j * g * (c * a.dag() - c.dag() * a))


def atom_in_irregular_cavity(I: qt.Qobj, a_list: List[qt.Qobj], c: qt.Qobj, gammas: List[float], Deltas: List[float],
                             Omega: float, gs: List[float]) -> nw.Component:
    """
    Creates an atom in a one-sided cavity, with many modes available for transition, so the Hamiltonian is
    H = Omega / 2 * sigma_z + sum_i Delta_i a_i^dag * a_i + g_i * (c * a_i^dag + c^dag * a_i) coupled to external
    degrees of freedom with rate gamma.
    :param I: The Identity operator on the full Hilbert space
    :param a_list: A list of the annihilation operators of the cavity in the full Hilbert space
    :param c: The sigma minus operator for the atom
    :param gammas: The decay rates of the cavity modes
    :param Deltas: The detunings of the cavity modes
    :param Omega: Atomic transition frequency
    :param gs: Coupling constants between cavity modes and atom
    :return: An SLH component for the atom in the cavity
    """
    sigma_z: qt.Qobj = c * c.dag() - c.dag() * c

    H = Omega / 2 * sigma_z
    L = 0
    for i in range(len(Deltas)):
        a_i = a_list[i]
        Delta_i = Deltas[i]
        g_i = gs[i]
        gamma_i = gammas[i]
        H += Delta_i * a_i.dag() * a_i + 1j * g_i * (a_i.dag() * c - c.dag() * a_i)
        L += np.sqrt(gamma_i) * a_i

    return nw.Component(S=nw.MatrixOperator(I), L=nw.MatrixOperator(L), H=H)


def virtual_cavity_in_irregular_cavity(I: qt.Qobj, a_list: List[qt.Qobj], c: qt.Qobj, gammas: List[float],
                                       Deltas: List[float], Omega: float,
                                       gs: List[Callable[[float], float]]) -> nw.Component:
    """
    Creates an atom in a one-sided cavity, with many modes available for transition, so the Hamiltonian is
    H = Omega / 2 * sigma_z + sum_i Delta_i a_i^dag * a_i + g_i * (c * a_i^dag + c^dag * a_i) coupled to external
    degrees of freedom with rate gamma.
    :param I: The Identity operator on the full Hilbert space
    :param a_list: A list of the annihilation operators of the cavity in the full Hilbert space
    :param c: The sigma minus operator for the atom
    :param gammas: The decay rates of the cavity modes
    :param Deltas: The detunings of the cavity modes
    :param Omega: Atomic transition frequency
    :param gs: Coupling constants between cavity modes and atom
    :return: An SLH component for the atom in the cavity
    """
    sigma_z: qt.Qobj = c * c.dag() - c.dag() * c

    H = Omega / 2 * sigma_z
    L = 0
    for i in range(len(Deltas)):
        a_i = a_list[i]
        Delta_i = Deltas[i]
        g_i = gs[i]
        gamma_i = gammas[i]
        c_t: qt.QobjEvo = qt.QobjEvo([[c, lambda t, args: np.conjugate(g_i(t))]])
        H += Delta_i * a_i.dag() * a_i + 0.5j * (a_i.dag() * c_t - c_t.dag() * a_i)
        L += np.sqrt(gamma_i) * a_i

    return nw.Component(S=nw.MatrixOperator(I), L=nw.MatrixOperator(L), H=H)


def two_virtual_cavities_in_irregular_cavity(I: qt.Qobj, a_list: List[qt.Qobj], c1: qt.Qobj, c2: qt.Qobj,
                                             gammas: List[float], Deltas: List[float], Omega: float,
                                             g1s: List[Callable[[float], float]],
                                             g2s: List[Callable[[float], float]]) -> nw.Component:
    """
    Creates an atom in a one-sided cavity, with many modes available for transition, so the Hamiltonian is
    H = Omega / 2 * sigma_z + sum_i Delta_i a_i^dag * a_i + g_i * (c * a_i^dag + c^dag * a_i) coupled to external
    degrees of freedom with rate gamma.
    :param I: The Identity operator on the full Hilbert space
    :param a_list: A list of the annihilation operators of the cavity in the full Hilbert space
    :param c: The sigma minus operator for the atom
    :param gammas: The decay rates of the cavity modes
    :param Deltas: The detunings of the cavity modes
    :param Omega: Atomic transition frequency
    :param gs: Coupling constants between cavity modes and atom
    :return: An SLH component for the atom in the cavity
    """
    H = Omega * c1.dag() * c1 + Omega * c2.dag() * c2
    L = 0
    for i in range(len(Deltas)):
        a_i = a_list[i]
        Delta_i = Deltas[i]
        g1_i = g1s[i]
        g2_i = g2s[i]
        gamma_i = gammas[i]
        c1_t: qt.QobjEvo = qt.QobjEvo([[c1, lambda t, args, g1_n=g1_i: np.conjugate(g1_n(t))]])
        c2_t: qt.QobjEvo = qt.QobjEvo([[c2, lambda t, args, g2_n=g2_i: np.conjugate(g2_n(t))]])
        H += Delta_i * a_i.dag() * a_i + 1j * (a_i.dag() * c1_t - c1_t.dag() * a_i + a_i.dag() * c2_t - c2_t.dag() * a_i)
        L += np.sqrt(gamma_i) * a_i

    return nw.Component(S=nw.MatrixOperator(I), L=nw.MatrixOperator(L), H=H)
