"""
Implements several physics functions often used
"""
import time
import sys
import qutip as qt
import numpy as np
from numpy import ndarray
from scipy.integrate import trapz
import SLH.network as nw
from util.quantumsystem import QuantumSystem
import util.plots as plots
from util.plots import SubPlotOptions, LineOptions
from typing import Callable, List, Any, Tuple, Optional, Union


def autocorrelation(liouvillian: Callable[[float, Any], qt.Qobj], psi: qt.Qobj, times: np.ndarray,
                    a_op: qt.QobjEvo, b_op: qt.QobjEvo) -> np.ndarray:
    """
    Calculates the autocorrelation function (eq. 14 in long Kiilerich) as a matrix of t and t'. The a_op and b_op must
    be time-dependent QobjEvo operators. For non-time dependent operators, use qutips own correlation functions
    :param liouvillian: The liouvillian to use for time evolution
    :param psi: The initial state
    :param times: An array of the times to evaluate the autocorrelation function
    :param a_op: The time-dependent rightmost operator in the autocorrelation function
    :param b_op: The time-dependent leftmost operator in the autocorrelation function
    :return: A matrix of the autocorrelation function evaluated at t and t'
    """
    result: qt.solver.Result = integrate_master_equation(f=liouvillian, psi=psi, c_ops=[], e_ops=[], times=times)
    rhos: np.ndarray = result.states
    autocorr_matrix = np.zeros((len(times), len(times)), dtype=np.complex_)

    def b_op_t(t: float, state) -> float:
        return qt.expect(b_op(t), state)

    for t_idx, rho in enumerate(rhos):
        sys.stdout.write("\r" + "Iteration " + str(t_idx) + " out of " + str(len(times)))
        ex = integrate_master_equation(f=liouvillian, psi=a_op(times[t_idx]) * rho, c_ops=[],
                                       e_ops=[b_op_t], times=times[t_idx:], verbose=False).expect[0]
        autocorr_matrix[t_idx, t_idx:] = ex
        autocorr_matrix[t_idx:, t_idx] = ex
        sys.stdout.flush()
    print("\n")
    return autocorr_matrix


def autocorrelation_test(liouvillian: Callable[[float, Any], qt.Qobj], psi: qt.Qobj, times: np.ndarray,
                         a_op: qt.QobjEvo, b_op: qt.Qobj, b_t: Callable[[float], float]):
    def a_op_t(t: float, state) -> float:
        return qt.expect(a_op(t), state)
    result: qt.solver.Result = integrate_master_equation(f=liouvillian, psi=qt.ket2dm(psi)*b_op, c_ops=[],
                                                         e_ops=[a_op_t], times=times)
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
                             trim: bool = False) -> Tuple[ndarray, List[float], List[np.ndarray]]:
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

    vals, vecs = convert_autocorr_mat_to_vals_and_vecs(autocorr_mat, times, n=n, trim=trim)

    return autocorr_mat, vals, vecs


def convert_autocorr_mat_to_vals_and_vecs(autocorr_mat, times: ndarray, n: Optional[int] = None,
                                          trim: bool = False) -> Tuple[List[float], List[np.ndarray]]:
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
    return vals, vecs


def integrate_master_equation(f: Union[qt.Qobj, qt.QobjEvo, Callable[[float, any], qt.Qobj]], psi: qt.Qobj,
                              c_ops: List[qt.Qobj], e_ops: List[qt.Qobj], times: np.ndarray,
                              options: qt.Options = qt.Options(nsteps=1000000000, store_states=1, atol=1e-8, rtol=1e-6),
                              verbose=True) -> qt.solver.Result:
    """
    Integrates the master equation for the system specifications specified in the setup.py file
    :param f: A  liouvillian object containing the Hamiltonian and Lindblad operators
    :param psi: The initial state as a ket
    :param c_ops: The collapse operators for the system
    :param e_ops: The observables to be tracked during the time-evolution
    :param times: An array of the times to evaluate the observables at
    :param options: The options for the integrator, as a qutip Options object
    :param verbose: Whether to display a progress bar or not. Default: True
    :return: The expectation values of the number operators for the ingoing pulse, outgoing pulse and system excitations
             in that order
    """
    if verbose:
        output = qt.mesolve(f, psi, tlist=times, c_ops=c_ops, e_ops=e_ops, progress_bar=True,
                            options=options)
    else:
        output = qt.mesolve(f, psi, tlist=times, c_ops=c_ops, e_ops=e_ops, options=options)
    return output


def calculate_expectations_and_states(system: nw.Component, psi: qt.Qobj,
                                      e_ops: List[Union[qt.Qobj, Callable[[float, Any], float]]],
                                      times, options, verbose=True) -> qt.solver.Result:
    """
    Calculates the expectation values and states at all times for a given SLH-component and some operators, by
    time-evolving the system Hamiltonian
    :param system: The SLH-component for the system
    :param psi: The initial state for the system
    :param e_ops: The operators to get expectation values of
    :param times: An array of the times to get the expectation values and the states
    :param options: The options for the integrator, as a qutip Options object
    :param verbose: Whether to display a progress bar or not. Default: True
    :return: A QuTiP result class, containing the expectation values and states at all times
    """
    print("Initializing simulation")
    t1 = time.time()
    if system.is_L_temp_dep():
        result = integrate_master_equation(system.liouvillian, psi, c_ops=[], e_ops=e_ops, times=times,
                                           options=options, verbose=verbose)
    else:
        H = system.H
        result = integrate_master_equation(H, psi, c_ops=system.get_Ls(), e_ops=e_ops, times=times, options=options,
                                           verbose=verbose)
    print(f"Finished in {time.time() - t1} seconds!")
    return result


def quantum_trajectory_method(H: Union[qt.Qobj, qt.QobjEvo],
                              Ls: List[Union[qt.Qobj, qt.QobjEvo]],
                              Ls_mon: List[Union[qt.Qobj, qt.QobjEvo]],
                              psi: qt.Qobj,
                              e_ops: List[Union[qt.Qobj, qt.QobjEvo]],
                              times: np.ndarray, n: int) -> List[qt.solver.Result]:
    """
    Performs the quantum trajectory method, where each loss of quantum content is accounted for (see Niels Munch
    Mikkelsen's Bachelor thesis)
    :param H: The Hamiltonian to time-evolve. Use QObjEvo if time-dependent
    :param Ls: List of the Lindblad loss terms not monitored. Use QObjEvo if time-dependent
    :param Ls_mon: List of the Lindblad loss terms that are monitored. These terms are monitored for how many quanta
                   decays through these channels Use QObjEvo if time-dependent
    :param psi: The initial state
    :param e_ops: A list of observables to take expectation value of at each time step. Use QObjEvo if time-dependent
    :param times: A list of the times to evaluate state and expectation values at
    :param n: The total number of lost quanta to be accounted for
    :return: A list of qutip Result objects for each n
    """
    if psi.isket:
        dm = qt.ket2dm(psi)  # Density matrix of initial state
    else:
        dm = psi

    time_dep_H = isinstance(H, qt.QobjEvo)
    time_dep_Ls = [False for _ in Ls]
    time_dep_Ls_mon = [False for _ in Ls_mon]
    for l, L in enumerate(Ls):
        time_dep_Ls[l] = isinstance(L, qt.QobjEvo)
    for l, L_mon in enumerate(Ls_mon):
        time_dep_Ls_mon[l] = isinstance(L_mon, qt.QobjEvo)

    T = times[-1]
    nT = len(times)
    dt = T/nT

    rho_ks = [0]
    results = []

    for i in range(n + 1):
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
            Lts = [L(t) if time_dep_Ls[l] else L for l, L in enumerate(Ls)]
            Lts_mon = [L_mon(t) if time_dep_Ls_mon[l] else L_mon for l, L_mon in enumerate(Ls_mon)]

            rho_k = rho_ks[-1]
            rho = rho_t[j]
            if isinstance(rho_k, list):
                rho_k = rho_k[j]

            out: qt.Qobj = 1j * qt.commutator(Ht, rho, 'normal')
            for Lt in Lts:
                out += 0.5 * qt.commutator(Lt.dag()*Lt, rho, 'anti')
                out -= Lt * rho * Lt.dag()
            for Lt_mon in Lts_mon:
                out += 0.5 * qt.commutator(Lt_mon.dag()*Lt_mon, rho, 'anti')
                out -= Lt_mon * rho_k * Lt_mon.dag()
            rho_t[j + 1] = rho_t[j] - dt*out

            for k, e in enumerate(e_ops):
                if isinstance(e, qt.QobjEvo):
                    e_ops_t[k][j] = qt.expect(rho_t[j], e(t))
                else:
                    e_ops_t[k][j] = qt.expect(rho_t[j], e)

        res.states = rho_t[0:nT]
        res.expect = e_ops_t
        rho_ks.append(rho_t)
        results.append(res)
    return results


"""Functions for running Quantum Systems"""


def run_quantum_system(quantum_system: QuantumSystem, plot: bool = True, verbose: bool = True) -> qt.solver.Result:
    """
    Runs an interferometer, with an SLH-component, pulse-shapes and initial states along with some defined operators
    to get expectation values of. Plots the final result
    :param quantum_system: The interferometer to time-evolve
    :param plot: Boolean of whether to plot the result
    :param verbose: Whether to display a progress bar or not. Default: True
    :return: A qutip Result-object with the result of the simulation
    """
    pulses = quantum_system.pulses
    psi0 = quantum_system.psi0
    times = quantum_system.times
    options = quantum_system.options
    pulse_options, content_options = quantum_system.get_plotting_options()
    total_system: nw.Component = quantum_system.create_component()
    e_ops: Union[List[qt.Qobj], Callable] = quantum_system.get_expectation_observables()

    # Test plotting options
    assert len(pulse_options) == len(pulses)
    if isinstance(e_ops, Callable):
        # TODO: Implement check of length of output from function
        pass
    else:
        assert len(e_ops) == len(content_options)

    result: qt.solver.Result = calculate_expectations_and_states(total_system, psi0, e_ops, times, options, verbose)

    if isinstance(e_ops, Callable):
        result.expect = convert_time_dependent_e_ops_list(result, times)

    if plot:
        plots.plot_system_contents(times, pulses, pulse_options, result.expect, content_options)

    return result


def run_autocorrelation(interferometer: QuantumSystem, n: int = 6, trim: bool = False)\
        -> Tuple[List[List[float]], List[List[np.ndarray]]]:
    """
    Calculates the autocorrelation functions on all output channels of an interferometer, to find the pulse modes and
    content of the pulse mode at each interferometer output
    :param interferometer: The interferometer to find the output from
    :param n: The number of most populated orthogonal output modes to produce
    :param trim: Whether to trim modes with less than 0.001 photons
    """
    psi0 = interferometer.psi0
    times = interferometer.times
    total_system: nw.Component = interferometer.create_component()

    Ls: List[qt.QobjEvo] = total_system.get_Ls()
    vals_in_arms: List[List[float]] = []
    vecs_in_arms: List[List[np.ndarray]] = []
    for L in Ls:
        autocorr_mat, vals, vecs = get_most_populated_modes(total_system.liouvillian, L, psi0, times, n=n, trim=trim)
        vals_in_arms.append(vals)
        vecs_in_arms.append(vecs)
        #with open(f"output_modes/exact_simple_interferometer_1_photons.pk1", "wb") as file:
        #    pickle.dump(vecs, file)
        plots.plot_autocorrelation(autocorr_mat=autocorr_mat, vs=vecs, eigs=vals, times=times)
    return vals_in_arms, vecs_in_arms


def run_quantum_trajectory(quantum_system: QuantumSystem, n: int, plot=False):
    """
    Finds the quantum trajectory for each number of quanta lost to output modes for each L in the component of the
    quantum system
    :param quantum_system: The quantum system to run the method on
    :param n: The total number of lost quanta to be accounted for
    :param plot: Whether to plot the result or not
    :return: A list of lists of the results, A list for each L containing a list for each lost quantum
    """
    total_system: nw.Component = quantum_system.create_component()
    Ls = total_system.get_Ls()
    all_results = []
    for l, L in enumerate(Ls):
        Ls_other = []
        for k in range(len(Ls)):
            if l != k:
                Ls_other.append(Ls[k])
        results: List[qt.solver.Result] = quantum_trajectory_method(total_system.H,
                                                                    Ls_other,
                                                                    [L],
                                                                    quantum_system.psi0,
                                                                    quantum_system.get_expectation_observables(),
                                                                    quantum_system.times,
                                                                    n)
        all_results.append(results)
        if plot:
            for i, result in enumerate(results):
                no_of_quanta = result.expect[-1][-1]
                print(f"Prob. of {i} number of quanta is {no_of_quanta}")
            xs_list = [[quantum_system.times for _ in res.expect] for res in results]
            ys_list = [result.expect for result in results]
            pulse_options, content_options = quantum_system.get_plotting_options()
            content_options_list = [content_options for _ in results]
            subplot_options_list = [SubPlotOptions(xlabel="times", ylabel=f"content {i}") for i in range(n+1)]
            plots.simple_subplots(xs_list, ys_list, content_options_list, subplot_options_list,
                                  title='quantum trajectory')
    return all_results


def run_multiple_quantum_trajectories(quantum_system: QuantumSystem, n: int,
                                      taus: np.ndarray, tps: np.ndarray, Ts: np.ndarray):
    nT = len(quantum_system.times)
    no_of_Ls = len(quantum_system.create_component().get_Ls())

    all_no_of_quanta = [[[] for _ in range(n + 1)] for _ in range(no_of_Ls)]
    for k in range(len(taus)):
        print(f"Iteration: {k}")
        tau = taus[k]
        tp = tps[k]
        T = Ts[k]
        quantum_system.redefine_pulse_args([tp, tau])
        quantum_system.times = np.linspace(0, T, nT)

        total_system: nw.Component = quantum_system.create_component()
        Ls = total_system.get_Ls()
        all_results = []
        for l, L in enumerate(Ls):
            Ls_other = []
            for j in range(len(Ls)):
                if l != j:
                    Ls_other.append(Ls[j])
            results: List[qt.solver.Result] = quantum_trajectory_method(total_system.H,
                                                                        Ls_other,
                                                                        [L],
                                                                        quantum_system.psi0,
                                                                        quantum_system.get_expectation_observables(),
                                                                        quantum_system.times,
                                                                        n)
            all_results.append(results)
            for i, result in enumerate(results):
                no_of_quanta = result.expect[-1][-1]
                all_no_of_quanta[l][i].append(no_of_quanta)
    xs_list = [[taus for _ in range(n + 1)] for _ in all_no_of_quanta]
    ys_list = [[no_of_quantas[i] for i in range(n + 1)] for no_of_quantas in all_no_of_quanta]
    content_options_list = [[LineOptions(linetype="-", linewidth=4, color="r", label=f"prob. 0 quanta"),
                             LineOptions(linetype=":", linewidth=4, color="g", label=f"prob. 1 quanta"),
                             LineOptions(linetype="--", linewidth=4, color="b", label=f"prob. 2 quanta")]
                            for _ in all_no_of_quanta]
    subplot_options_list = [SubPlotOptions(xlabel="taus", ylabel=f"arm {i}") for i in range(len(all_no_of_quanta))]
    plots.simple_subplots(xs_list, ys_list, content_options_list, subplot_options_list,
                          title='quantum trajectory for each arm')


def run_multiple_tau(interferometer: QuantumSystem, taus: np.ndarray, tps: np.ndarray, Ts: np.ndarray):
    """
    Gets the photon-population at each interferometer arm as a function of pulse length tau and plots the result
    :param interferometer: The interferometer to time-evolve
    :param taus: An array of the taus to evaluate the photon population of
    :param tps: A corresponding array of pulse delays, such that the gaussian pulse is contained within t = 0:T
    :param Ts: A corresponding array of max times, such that the gaussian pulse is contained within t = 0:T
    """
    psi0 = interferometer.psi0
    options = interferometer.options

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

        result: qt.solver.Result = calculate_expectations_and_states(total_system, psi0, e_ops, times, options)
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
    options = interferometer.options
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

        result: qt.solver.Result = calculate_expectations_and_states(total_system, psi0, e_ops, times, options)
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


"""Helper functions"""


def convert_time_dependent_e_ops_list(result: qt.solver.Result, times: np.ndarray) -> List[List]:
    nT = len(times)
    expect = [[0 for _ in range(nT)] for _ in range(len(result.expect[0]))]

    for i, t in enumerate(times):
        for j in range(len(result.expect[0])):
            expect[j][i] = result.expect[i][j]

    return expect
