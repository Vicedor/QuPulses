"""
Implements several physics functions often used
"""
import time
import qutip as qt
import numpy as np
from numpy import ndarray
from scipy.integrate import trapz
import SLH.network as nw
from util.quantumsystem import QuantumSystem
import util.plots as plots
from typing import Callable, List, Any, Tuple, Optional, Union


def autocorrelation(liouvillian: Callable[[float, Any], qt.Qobj], psi: qt.Qobj, times: np.ndarray,
                    a_op: qt.QobjEvo, b_op: qt.QobjEvo) -> np.ndarray:
    """
    Calculates the autocorrelation function (eq. 14 in long Kiilerich) as a matrix of t and t'. The a_op and b_op must
    be time dependent QobjEvo operators. For non-time dependent operators, use qutips own correlation functions
    :param liouvillian: The liouvillian to use for time evolution
    :param psi: The initial state
    :param times: An array of the times to evaluate the autocorrelation function
    :param a_op: The time-dependent rightmost operator in the autocorrelation function
    :param b_op: The time-dependent leftmost operator in the autocorrelation function
    :return: A matrix of the autocorrelation function evaluated at t and t'
    """
    result: qt.solver.Result = integrate_master_equation(liouvillian=liouvillian, psi=psi, e_ops=[], times=times)
    rhos: np.ndarray = result.states
    autocorr_matrix = np.zeros((len(times), len(times)), dtype=np.complex_)

    def b_op_t(t: float, state) -> float:
        return qt.expect(b_op(t), state)

    for t_idx, rho in enumerate(rhos):
        ex = integrate_master_equation(liouvillian=liouvillian, psi=a_op(times[t_idx]) * rho,
                                       e_ops=[b_op_t], times=times[t_idx:]).expect[0]
        autocorr_matrix[t_idx, t_idx:] = ex
        autocorr_matrix[t_idx:, t_idx] = ex
    return autocorr_matrix


def autocorrelation_test(liouvillian: Callable[[float, Any], qt.Qobj], psi: qt.Qobj, times: np.ndarray,
                         a_op: qt.QobjEvo, b_op: qt.Qobj, b_t: Callable[[float], float]):
    def a_op_t(t: float, state) -> float:
        return qt.expect(a_op(t), state)
    result: qt.solver.Result = integrate_master_equation(liouvillian=liouvillian, psi=qt.ket2dm(psi)*b_op, e_ops=[a_op_t], times=times)
    ex = result.expect[0]
    autocorr_matrix = np.zeros((len(times), len(times)), dtype=np.complex_)
    for t_idx1, t1 in enumerate(times):
        for t_idx2, t2 in enumerate(times):
            if t_idx1 <= t_idx2:
                res = ex[t_idx1] * b_t(t2)
                autocorr_matrix[t_idx1, t_idx2] = res
                autocorr_matrix[t_idx2, t_idx1] = res
    return autocorr_matrix


def convert_correlation_output(vec: np.ndarray, val: complex, times: np.ndarray) -> Tuple[np.ndarray, complex]:
    """
    Converts an eigenvector and eigenvalue from the output of the autocorrelation function, to normalize the eigenvalues
    to the total number of photons, and to normalize the eigenvector
    :param vec: The eigenvector from the autocorrelation function output
    :param val: The corresponding eigenvalue from the autocorrelation function output
    :param times: The array of times at which the eigenvector is evaluated
    :return: A normalized eigenvector and eigenvalue
    """
    vec_int = trapz(vec * np.conjugate(vec), times)
    val1 = val * vec_int
    vec1 = vec / np.sqrt(vec_int)
    return vec1, val1


def get_autocorrelation_function(liouvillian: Callable[[float, Any], qt.Qobj], psi: qt.Qobj, a_op: qt.QobjEvo,
                                 b_op: qt.QobjEvo, times: np.ndarray) -> np.ndarray:
    """
    Calculates the autocorrelation function given a system liouvillian, an initial state and the two system operators in
    the autocorrelation function
    :param liouvillian: The system liouvillian for time-evolution
    :param psi: The initial state for the system
    :param a_op: The time-dependent rightmost operator in the autocorrelation function
    :param b_op: The time-dependent leftmost operator in the autocorrelation function
    :param times: An array of the times to evaluate the autocorrelation function
    :return: The matrix of autocorrelation function values (size times x times)
    """
    print("Starting to calculate autocorrelation function")
    t2 = time.time()
    autocorr_mat = autocorrelation(liouvillian, psi, times, a_op=a_op, b_op=b_op)
    print(f"Finished in {time.time() - t2} seconds!")
    return autocorr_mat


def get_most_populated_modes(liouvillian: Callable[[float, Any], qt.Qobj], L: qt.QobjEvo, psi0: qt.Qobj, times: ndarray,
                             n: Optional[int] = None,
                             trim: bool = False) -> Tuple[ndarray, List[Any], List[Tuple[ndarray, complex]]]:
    """
    Finds the most populated modes from the autocorrelation function. First the autocorrelation matrix is calculated,
    then it is diagonalized into eigenvalues and eigenvectors. The eigenvectors with the largest eigenvalues
    (corresponding to the most populated modes) are found.
    :param liouvillian: The system liouvillian for time-evolution of the system
    :param L: The time-dependent Lindblad operator for the system loss
    :param psi0: The initial state of the system
    :param times: The timesteps to evaluate the autocorrelation function
    :param n: The number of modes to retrieve (if None all modes with more than 0.001 photon content is found, though
              a maximum of 10 modes are found)
    :param trim: Boolean value for whether to trim modes with less than 0.001 photon content
    :return: The autocorrelation matrix, eigenvalues and eigenvectors for the most populated modes
    """
    autocorr_mat: np.ndarray = get_autocorrelation_function(liouvillian, psi0, a_op=L, b_op=L.dag(), times=times)

    val, vec = np.linalg.eig(autocorr_mat)

    if n is None:
        n = 10
        trim = True

    vecs = [vec[:, i] for i in range(n)]
    vals = [val[i] for i in range(n)]
    for i, v in enumerate(vecs):
        vecs[i], vals[i] = convert_correlation_output(v, vals[i], times)

    if trim:
        vecs2 = []
        vals2 = []
        for i, v in enumerate(vals):
            if v > 0.001:
                vecs2.append(vecs[i])
                vals2.append(v)
        vecs = vecs2
        vals = vals2

    return autocorr_mat, vals, vecs


def integrate_master_equation(liouvillian: Callable[[float, any], qt.Qobj], psi: qt.Qobj,
                              e_ops: List[qt.Qobj], times: np.ndarray) -> qt.solver.Result:
    """
    Integrates the master equation for the system specifications specified in the setup.py file
    :param liouvillian: A liouvillian object containing the Hamiltonian and Lindblad operators
    :param psi: The initial state as a ket
    :param e_ops: The observables to be tracked during the time-evolution
    :param times: An array of the times to evaluate the observables at
    :return: The expectation values of the number operators for the ingoing pulse, outgoing pulse and system excitations
             in that order
    """
    if psi.isket:
        dm = qt.ket2dm(psi)  # Density matrix of initial state
    else:
        dm = psi
    output = qt.mesolve(liouvillian, dm, tlist=times, c_ops=[], e_ops=e_ops,
                        options=qt.Options(nsteps=1000000000, store_states=1, atol=1e-8, rtol=1e-6))
    return output


def calculate_expectations_and_states(system: nw.Component, psi: qt.Qobj,
                                      e_ops: List[Union[qt.Qobj, Callable[[float, Any], float]]],
                                      times) -> qt.solver.Result:
    """
    Calculates the expectation values and states at all times for a given SLH-component and some operators, by
    time-evolving the system Hamiltonian
    :param system: The SLH-component for the system
    :param psi: The initial state for the system
    :param e_ops: The operators to get expectation values of
    :param times: An array of the times to get the expectation values and the states
    :return: A QuTiP result class, containing the expectation values and states at all times
    """
    print("Initializing simulation")
    t1 = time.time()
    result = integrate_master_equation(system.liouvillian, psi, e_ops, times)
    print(f"Finished in {time.time() - t1} seconds!")
    return result


"""Functions for running interferometers"""


def run_interferometer(interferometer: QuantumSystem, plot: bool = True) -> qt.solver.Result:
    """
    Runs an interferometer, with an SLH-component, pulse-shapes and initial states along with some defined operators
    to get expectation values of. Plots the final result
    :param interferometer: The interferometer to time-evolve
    :param plot: Boolean of whether to plot the result
    """
    pulses = interferometer.pulses
    psi0 = interferometer.psi0
    times = interferometer.times
    pulse_options, content_options = interferometer.get_plotting_options()
    total_system: nw.Component = interferometer.create_component()
    e_ops: List[qt.Qobj] = interferometer.get_expectation_observables()

    # Test plotting options
    assert len(pulse_options) == len(pulses)
    assert len(e_ops) == len(content_options)

    result: qt.solver.Result = calculate_expectations_and_states(total_system, psi0, e_ops, times)

    if plot:
        plots.plot_system_contents(times, pulses, pulse_options, result.expect, content_options)

    return result


def run_autocorrelation(interferometer: QuantumSystem):
    """
    Calculates the autocorrelation functions on all output channels of an interferometer, to find the pulse modes and
    content of the pulse mode at each interferometer output
    :param interferometer: The interferometer to find the output from
    """
    psi0 = interferometer.psi0
    times = interferometer.times
    total_system: nw.Component = interferometer.create_component()

    Ls: List[qt.QobjEvo] = total_system.get_Ls()
    for L in Ls:
        autocorr_mat, vals, vecs = get_most_populated_modes(total_system.liouvillian, L, psi0, times, n=6)
        #with open(f"output_modes/exact_simple_interferometer_1_photons.pk1", "wb") as file:
        #    pickle.dump(vecs, file)
        plots.plot_autocorrelation(autocorr_mat=autocorr_mat, vs=vecs, eigs=vals, times=times)


def run_multiple_tau(interferometer: QuantumSystem, taus: np.ndarray, tps: np.ndarray, Ts: np.ndarray):
    """
    Gets the photon-population at each interferometer arm as a function of pulse length tau and plots the result
    :param interferometer: The interferometer to time-evolve
    :param taus: An array of the taus to evaluate the photon population of
    :param tps: A corresponding array of pulse delays, such that the gaussian pulse is contained within t = 0:T
    :param Ts: A corresponding array of max times, such that the gaussian pulse is contained within t = 0:T
    """
    psi0 = interferometer.psi0

    arm0_populations = []
    arm1_populations = []

    for i in range(len(taus)):
        T = Ts[i]
        tau = taus[i]
        nT = 500  # 1000                           # The number of points in time to include
        times = np.linspace(0, T, nT)  # The list of points in time to evaluate the observables
        tp = tps[i]  # 4

        interferometer.redefine_pulse_args([tp, tau])
        total_system: nw.Component = interferometer.create_component()

        L0: qt.QobjEvo = total_system.get_Ls()[0]
        L1: qt.QobjEvo = total_system.get_Ls()[1]

        def L0dagL0t(t: float, state) -> float:
            return qt.expect(L0(t).dag() * L0(t), state)

        def L1dagL1t(t: float, state) -> float:
            return qt.expect(L1(t).dag() * L1(t), state)

        e_ops = [L0dagL0t, L1dagL1t]

        result: qt.solver.Result = calculate_expectations_and_states(total_system, psi0, e_ops, times)
        arm0_population_t, arm1_population_t = result.expect

        arm0_population = sum(arm0_population_t) * (T / nT)  # integrate over L0dagL0
        arm1_population = sum(arm1_population_t) * (T / nT)  # integrate over L1dagL1

        arm0_populations.append(arm0_population)
        arm1_populations.append(arm1_population)

    plots.plot_arm_populations(taus, arm0_populations, arm1_populations)


def run_optimize_squeezed_states(interferometer: QuantumSystem, N: int):
    xis = np.linspace(0.1, 2, 40)
    arm0_populations = []
    arm1_populations = []
    psi0s = qt.basis(2, 0)  # Initial system state    for xi in xis:

    times = interferometer.times
    T = times[-1]
    nT = len(times)

    for xi in xis:
        psi0u, success_prob = get_photon_subtracted_squeezed_state(N, xi)
        psi0 = qt.tensor(psi0u, psi0s)
        input_photons = qt.expect(qt.create(N) * qt.destroy(N), psi0u)
        total_system: nw.Component = interferometer.create_component()
        L0: qt.QobjEvo = total_system.get_Ls()[0]
        L1: qt.QobjEvo = total_system.get_Ls()[1]

        def L0dagL0t(t: float, state) -> float:
            return qt.expect(L0(t).dag() * L0(t), state)

        def L1dagL1t(t: float, state) -> float:
            return qt.expect(L1(t).dag() * L1(t), state)

        e_ops = [L0dagL0t, L1dagL1t]

        result: qt.solver.Result = calculate_expectations_and_states(total_system, psi0, e_ops, times)
        arm0_population_t, arm1_population_t = result.expect

        arm0_population = sum(arm0_population_t) * (T / nT) / input_photons  # integrate over L0dagL0
        arm1_population = sum(arm1_population_t) * (T / nT) / input_photons * success_prob  # integrate over L1dagL1

        arm0_populations.append(arm0_population)
        arm1_populations.append(arm1_population)
    plots.plot_arm_populations(xis, arm0_populations, arm1_populations)


"""Functions for getting different kinds of states"""


def get_photon_subtracted_squeezed_state(N: int, xi: complex) -> Tuple[qt.Qobj, float]:
    """
    Gets the normalized photon subtracted squeezed state
    :param N: The size of the Hilbert space
    :param xi: The xi-parameter for the squeezed state
    :return: The photon subtracted squeezed state as a Qobj and the success probability of creating it
    """
    squeezed_state = qt.squeeze(N, xi) * qt.basis(N, 0)
    success_prob = 1 - qt.expect(qt.ket2dm(qt.basis(N, 0)), squeezed_state)
    photon_subtracted_squeezed_state = qt.destroy(N) * squeezed_state
    return photon_subtracted_squeezed_state.unit(), success_prob


def get_odd_schrodinger_cat_state(N: int, alpha: complex) -> qt.Qobj:
    """
    Generates an odd schrödinger cat state of the form given in eq. 7.116 in Gerry and Knight, Introductory Quantum
    optics
    :param N: The size of the Hilbert space
    :param alpha: The alpha coefficient for the coherent state
    :return: The Qobj for the odd cat state
    """
    odd_cat_state: qt.Qobj = (qt.coherent(N, alpha) - qt.coherent(N, -alpha)) / np.sqrt(2*(1 - np.exp(-2*alpha**2)))
    return odd_cat_state


"""
THE TWO FOLLOWING FUNCTION ARE NOT YET FINISHED!

Especially the final function, as it does not work
"""


def get_closest_index(t, times):
    for i, time in enumerate(times):
        if t <= time:
            return i
    return len(times) - 1


def quantum_trajectory_method(H: Union[qt.Qobj, qt.QobjEvo], L: Union[qt.Qobj, qt.QobjEvo], psi: qt.Qobj,
                              e_ops: List[qt.Qobj], times: np.ndarray, n: int) -> List[qt.solver.Result]:
    if psi.isket:
        dm = qt.ket2dm(psi)  # Density matrix of initial state
    else:
        dm = psi

    time_dep_H = isinstance(H, qt.QobjEvo)
    time_dep_L = isinstance(L, qt.QobjEvo)

    T = times[-1]
    nT = len(times)
    dt = T/nT

    rho_ks = [0]
    results = []

    for i in range(n):
        print(f"Starting {i} iteration")
        res = qt.solver.Result()
        res.times = times
        rho_t = [0 for _ in range(nT + 1)]
        if i == 0:
            rho_t[0] = dm
        else:
            rho_t[0] = dm*0
        e_ops_t = [[None for _ in range(nT)] for _ in e_ops]

        for j, t in enumerate(times):
            if time_dep_H:
                Ht: qt.Qobj = H(t)
            else:
                Ht: qt.Qobj = H
            if time_dep_L:
                Lt: qt.Qobj = L(t)
            else:
                Lt = L
            LdagLt: qt.Qobj = Lt.dag() * Lt

            rho_k = rho_ks[-1]
            rho = rho_t[j]
            if isinstance(rho_k, list):
                rho_k = rho_k[j]

            const_term: qt.Qobj = Lt * rho_k * Lt.dag()
            rho_t[j + 1] = rho_t[j] - dt*(1j * qt.commutator(Ht, rho, 'normal')
                                          + 0.5 * qt.commutator(LdagLt, rho, 'anti')
                                          - const_term)

            for k, e in enumerate(e_ops):
                if isinstance(e, qt.QobjEvo):
                    e_ops_t[k][j] = qt.expect(rho_t[j], e(t))
                else:
                    e_ops_t[k][j]= qt.expect(rho_t[j], e)

        res.states = rho_t[0:nT]
        res.expect = e_ops_t
        rho_ks.append(rho_t)
        results.append(res)
    return results
