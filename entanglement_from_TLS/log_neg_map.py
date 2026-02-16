"""
log_neg_map.py

Computes a map of the logarithmic negativity for pairs of modes in the output radiation
of a driven two-level system (TLS). The output modes are found either through an eigenvalue
decomposition of the autocorrelation function g1(t, t') = <a_out^dagger(t) a_out(t')> or
through a Takagi decomposition of the correlation function f1(t, t') = <a_out(t) a_out(t')>

The output modes are used as a basis for the output radiation, with the goal of analyzing
the entanglement properties of the output radiation. To do so, we compute the quantum state
in pairs of output modes v_i(t) and v_j(t) using the virtual cavity method [1-3] to find
the quantum state in the output modes. We use the SLH formalism [4] to compute the
Hamiltonian and Lindblad terms for the quantum network consisting of the TLS followed
by the two virtual cavities in series.

The entanglement properties are quantified by the logarithmic negativity which is calculated
for each pair of modes for the 7 most populated modes. The maximum entanglement between
these 7 pairs are recorded as the entanglement for that set of parameters. The parameters
we vary are the detuning Delta, the pump strength Omega and the interaction length T.

We include two methods, one which explores the parameter space Omega, T for a given h.
This analysis shows that the maximum entanglement occurs along the ridge of the first
peak in the logarithmic negativity map. Therefore, we use a Brent optimization routine
to find the peak of this ridge, and record the Omega, T value for the peak for each h.

The method is an implementation of the work in {insert arXiv or journal reference}.

[1] A. H. Kiilerich and K. Mølmer, Input-output theory with quantum pulses, Phys. Rev. Lett. 123, 123604 (2019).
[2] A. H. Kiilerich and K. Mølmer, Quantum interactions with pulses of radiation, Phys. Rev. A 102, 023717 (2020).
[3] V. R. Christiansen, M. M. Lund, F. Yang, and K. Mølmer, Jaynes-cummings interaction with a traveling light pulse,
J. Opt. Soc. Am. B 41, C140 (2024).
[4] J. Combes, J. Kerckhoff, and M. Sarovar, The SLH framework for modeling quantum input-output networks,
Advances in Physics: X 2, 784 (2017).
"""

import time

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import pickle
import os
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from util.physics_functions import run_autocorrelation
from util.physics_functions import run_takagi
import SLH.network as nw
import SLH.component_factory as cf
import util.physics_functions as ph
import util.pulse as p
from util.quantumsystem import QuantumSystem
from util.plots import LineOptions

from typing import Any, Callable, Union, List, Tuple, Dict


# Use eigenvalue decomposition of g1 or Takagi decomposition of f1
decomp = 'eigval'
#decomp = 'takagi'

# Use Brent method or not
use_brent = False

# Overwrite previous results or not
do_overwrite = False

# The file for saving the result
eigval_file = r'eigval_log_neg_map.pkl'
takagi_file = r'takagi_log_neg_map.pkl'

eigval_brent_file = r'eigval_log_neg_map_brent.pkl'
takagi_brent_file = r'takagi_log_neg_map_brent.pkl'

# Dimension of the TLS
D = 2

# Tolerances
atol = 1e-2
rtol = 1e-2

# Size of the time grid for the decomposition
nT = 1000

# The decay rate of the TLS set to 1 as scale for the other variables
gamma = 1


def main():
    # The parameter h which determines Omega and Delta
    hs = np.linspace(0.1, 1, 10)

    for h in hs:
        print(f'calculating h={h}')
        run_log_neg_map(h, overwrite=do_overwrite)


def run_log_neg_map(h: float, overwrite: bool = False):
    """
    Runs a logarithmic entanglement map for a given h. Depending on the tag in the top of the file, it either uses the
    eigenvalue decomposition or Takagi decomposition to find the output modes. Also runs either a map for all OmegaRs
    and Ts, or use the Brent method to find the peak and the T, OmegaR pair that maximizes the logarithmic entanglement
    across the peak.

    Parameters
    ----------
    h : float
        The h value for the map
    overwrite : bool
        Whether to overwrite files for h-values that has already been calculated
    """
    # Find the file and task arguments depending on the decomposition and the method
    if decomp == 'eigval':
        task_args = (h, run_autocorrelation, get_hilbert_space_eigval)
        if use_brent:
            file = eigval_brent_file
        else:
            file = eigval_file
    elif decomp == 'takagi':
        task_args = (h, run_takagi, get_hilbert_space_takagi)
        if use_brent:
            file = takagi_brent_file
        else:
            file = takagi_file
    else:
        raise NotImplementedError

    # Check whether the file already exists - if so add the h-value map to the file
    if os.path.exists(file):
        with open(file, 'rb') as f:
            dic = pickle.load(f)
    else:
        dic = {}

    # If h already exists and overwrite is not toggled, return immediately
    if h in dic.keys():
        if not overwrite:
            return

    # Run the calculation of the map depending on the chosen method
    t1 = time.time()
    if use_brent:
        save_dic = log_neg_map_brent(h, task_args)
    else:
        save_dic = log_neg_map(h, task_args)
    t2 = time.time()

    # Add the map to the save file dictionary
    dic[h] = save_dic

    # Save the file with the new map
    with open(file, 'wb') as f:
        pickle.dump(dic, f)
    print(f'Finished in {t2 - t1} seconds')


def log_neg_map(task_args: Tuple[float, Callable, Callable]) -> Dict:
    """
    Computes a map of the logarithmic negativity as a function of the effective drive Omega_R and the pump duration T,
    for a given value of h. The result is saved in a dictionary along with the grid spacings for the parameters.

    The task arguments determine whether the map is computed using the eigenvalue decomposition of the g1(t, t')
    correlation function or the Takagi decomposition of the f1(t, t') decomposition.

    Parameters
    ----------
    task_args: tuple
        A tuple of the arguments for the map, consisting of h, the correlation function, and the function
        that generates the required Hilbert space dimensions.

    Returns
    -------
    dict
        A dictionary with the grid spacings OmegaRs and Ts, the value of h, and the logarithmic negativity map.
    """
    # Define the parameters for the map
    h = task_args[0]
    OmegaRs = np.linspace(0.1, 10, 50)
    Ts = np.linspace(0.1, 10, 50)

    # A blank array to keep the output
    max_log_negativity = np.zeros((len(OmegaRs), len(Ts)))

    # Record its shape to reshape later
    shape = max_log_negativity.shape

    # Create the value pairs for the parallel map
    values = []
    for OmegaR in OmegaRs:
        for T in Ts:
            values.append((T, OmegaR))

    # Run the function that calculates entries in the logarithmic entanglement map in parallel
    max_log_negativity = qt.parallel_map(calculate_log_negativity, values, task_args=task_args, progress_bar='enhanced')
    max_log_negativity = np.array(max_log_negativity).reshape(shape)

    # Create a dictionary of the parameters and the resulting map
    save_dic = {'OmegaRs': OmegaRs, 'Ts': Ts, 'h': h, 'max_log_negativity': max_log_negativity}

    # Plot the map
    plt.pcolormesh(Ts, OmegaRs, max_log_negativity)
    plt.xlabel('Integration time')
    plt.ylabel(r'$\Omega_R$')
    plt.title(f'Max log. negativity for $h$ = {h:.2f}')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return save_dic


def log_neg_map_brent(task_args):
    """
    Computes the maximum logarithmic negativity of the first peak in the T, OmegaR map, by keeping the value of
    T fixed and using a Brent maximization routine to find the OmegaR value that returns the largest logarithmic
    negativity. Record both the OmegaR value and the logarithmic negativity of this point.

    The task arguments determine whether the map is computed using the eigenvalue decomposition of the g1(t, t')
    correlation function or the Takagi decomposition of the f1(t, t') decomposition.

    Parameters
    ----------
    task_args: tuple
        A tuple of the arguments for the map, consisting of h, the correlation function, and the function
        that generates the required Hilbert space dimensions.

    Returns
    -------
    dict
        A dictionary with the grid spacings OmegaRs and Ts, the value of h, and the logarithmic negativity map.
    """
    # Define the parameters
    h = task_args[0]
    Ts = np.pi / np.linspace(np.pi / 2.5, np.pi / 10, 50)

    # Define the logarithmic negativity array
    max_log_negativity = np.zeros(Ts.shape)

    # Define the OmegaRs array which generate the maximum logarithmic negativity
    OmegaRs = np.zeros(Ts.shape)

    # Compute the logarithmic negativity and OmegaR pair for each T in parallel
    combined = qt.parallel_map(calculate_log_negativity_brent, Ts, task_args=task_args, progress_bar='enhanced')

    # Separate the OmegaR and logarithmic negativity pair
    for i, val in enumerate(combined):
        OmegaR, log_negativity = val
        max_log_negativity[i] = log_negativity
        OmegaRs[i] = OmegaR

    # Compute the save dictionary with the parameters and maximum logarithmic negativity
    save_dic = {'OmegaRs': OmegaRs, 'Ts': Ts, 'h': h, 'max_log_negativity': max_log_negativity}

    # Plot the maximum logarithmic negativity
    plt.figure()
    plt.plot(OmegaRs, max_log_negativity, label=f'$h={h:.2f}$')
    plt.xlabel(r'$\Omega_R$')
    plt.ylabel(r'Log. neg.')
    plt.title(f'Max log. negativity')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the OmegaR, T pair that maximizes the logarithmic negativity
    plt.figure()
    plt.plot(Ts, OmegaRs, label=f'$h={h:.2f}$')
    plt.xlabel(r'T')
    plt.ylabel(r'$\Omega_R$')
    plt.title(f'Best $\Omega_R$ as function of T')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return save_dic


def logarithmic_negativity(rho: qt.Qobj) -> float:
    """
    Computes the logarithmic negativity which is defined as the trace norm of the partial transpose, and then takes
    the base 2 logarithm of the result.

    Parameters
    ----------
    rho: qt.Qobj
        The qutip bipartite quantum state to compute the logarithmic negativity of.

    Returns
    -------
    float
        The logarithmic negativity of the input quantum state.
    """
    return np.log2(qt.partial_transpose(rho, [0, 1]).norm('tr'))


def convert_enr_to_dense(rho: qt.Qobj, Ns: List[int], excitations: int) -> qt.Qobj:
    """
    Converts an energy restricted quantum state to a dense full Hilbert space quantum state. The conversion is
    necessary since the inherent qutip functions, such as the partial transpose, only accepts full Hilbert space
    states.

    Parameters
    ----------
    rho: qt.Qobj
        The quantum object of an energy restricted Hilbert space dimension to convert to a dense full Hilbert space.
    Ns: list[int]
        The dimensions used to create the energy restricted Hilbert space of rho.
    excitations: int
        The maximum number of excitations for the energy restricted Hilbert space of rho.

    Returns
    -------
    qt.Qobj
        The quantum object of the input quantum state with the new dense full Hilbert space dimensions.
    """
    # Get the state to index map for the energy restricted Hilbert space
    out = qt.enr_state_dictionaries(Ns, excitations)
    state2idx: dict = out[1]

    # Get the state to index map for the full Hilbert space
    out_full = qt.enr_state_dictionaries(Ns, sum(Ns))
    state2idx_full: dict = out_full[1]

    # Get the quantum state array of the energy restricted state
    old_rho = rho.full()

    # Create a quantum state array to hold the full Hilbert space representation
    new_rho = np.zeros((np.prod(Ns), np.prod(Ns)), dtype=np.complex128)

    # Copy the state representation between the array representations
    for state1, idx1 in state2idx.items():
        for state2, idx2 in state2idx.items():
            c = old_rho[idx2, idx1]
            new_rho[state2idx_full[state2], state2idx_full[state1]] = c

    # Return a qutip Qobj with the new dimensions
    return qt.Qobj(new_rho, dims=[Ns, Ns])


def get_hilbert_space_eigval(T: float, Omega: float) -> Tuple[List[int], int]:
    """
    Find the Hilbert space dimensions for the eigenvalue decomposition of g1(t, t') output modes, for a given set
    of parameters. The Hilbert space dimensions are found by trial and error for a given set of parameters.

    Parameters
    ----------
    T: float
        The pump duration.
    Omega: float
        The pump strength.

    Return
    ------
    tuple[list[int], int]
        The dimensions of the output modes and the maximum number of excitations for the energy restricted Hilbert
        space dimensions.
    """
    # The dimension for the Hilbert space for the two modes. As T and Omega increases, a larger Hilbert space is needed.
    Ns_list = [[D, 7, 6], [D, 8, 7], [D, 10, 8], [D, 11, 10], [D, 10, 8]]
    # The number of necessary excitations for the energy restricted Hilbert space
    excitations_list = [7, 8, 10, 11, 9]

    # Find the necessary dimension for the given parameters
    if T <= 5:
        if Omega <= 5:
            k = 0
        else:
            k = 1
    elif T <= 7.5:
        if Omega <= 5:
            k = 4
        else:
            k = 3
    else:
        if Omega <= 2.5:
            k = 4
        elif Omega <= 5:
            k = 2
        else:
            k = 3

    return Ns_list[k], excitations_list[k]


def get_hilbert_space_takagi(T: float, Omega: float) -> Tuple[List[int], int]:
    """
    Find the Hilbert space dimensions for the Takagi decomposition of f1(t, t') output modes, for a given set
    of parameters. The Hilbert space dimensions are found by trial and error for a given set of parameters.

    Parameters
    ----------
    T: float
        The pump duration.
    Omega: float
        The pump strength.

    Return
    ------
    tuple[list[int], int]
        The dimensions of the output modes and the maximum number of excitations for the energy restricted Hilbert
        space dimensions.
    """
    Ns_list = [[D, 11, 10]]
    excitations_list = [11]
    return Ns_list[0], excitations_list[0]


def calculate_log_negativity(value: Tuple[float, float], *args) -> float:
    """
    Calculates the logarithmic negativity for the given set of parameters by calling the logneg function which is
    shared by the full map, which this function calculates, and the Brent method.

    Parameters
    ----------
    value: Tuple[float, float]
        The T, OmegaR pair for which the logarithmic negativity is calculated.
    args: Tuple[float, callable, callable]
        The value of h for which the map is calculated, the correlation function for the decomposition, and the
        Hilbert space function that generates the necessary Hilbert space dimensions.

    Returns
    -------
    float
        The maximum logarithmic negativity between a pair of modes for the given set of parameters.
    """
    T, OmegaR = value
    h, correlation_func, hilbert_space_func = args
    return - logneg(OmegaR, h, T, correlation_func, hilbert_space_func)


def calculate_log_negativity_brent(value: Tuple[float, float], *args)  -> Tuple[float, float]:
    """
    Calculates the logarithmic negativity for a given h, T value, where the Brent method is used to find the
    maximum logarithmic negativity over the OmegaR parameter.

    Parameters
    ----------
    value: Tuple[float, float]
        The h parameter and T parameter for the given calculation.
    args:
        The correlation function for the decomposition, and the Hilbert space function that generates the necessary
        Hilbert space dimensions.

    Returns
    -------
    tuple[float, float]
        The OmegaR that maximizes the logarithmic negativity, and the maximum logarithmic negativity.
    """
    h, T = value
    correlation_func, hilbert_space_func = args

    # Try to do the Brent routine using the minimize_scalar function from SciPy
    try:
        res = minimize_scalar(logneg, bracket=(0.5*np.pi / T, 1.5 * np.pi / T, 2.5 * np.pi / T),
                              tol=0.01, options={'maxiter': 20}, args=(h, T, correlation_func, hilbert_space_func))
    # If the routine fails, due to misplaced bracket or other errors, we catch the error to avoid the whole parallel
    # map failing, but instead return (0, 0) which we can replace later.
    except ValueError:
        print(f'Value Error at T = {T} and h = {h}')
        return 0, 0

    # Check if the routine succeeded. If so, return the found values
    if res.success:
        return res.x, -res.fun
    # If not, return the error values (0, 0)
    else:
        return 0, 0


def logneg(OmegaR: float, *args) -> float:
    """
    Computes the logarithmic negativity for a given OmegaR, T and h parameter set. The logarithmic negativity is found
    by first computing the output modes by calculating the chosen correlation function, and performing the relevant
    decomposition. The top 7 most populated modes are kept and used for calculating the joint quantum state of pairs
    of output modes. For each pair of output modes, their logarithmic negativity is found, and kept in an array.
    The largest of these values are returned as the maximum logarithmic negativity.

    Parameters
    ----------
    OmegaR: float
        The OmegaR parameter for the given calculation.
    args:
        The h, and T parameters for the calculation, and the chosen correlation function and the function that
        generates the Hilbert space dimensions for the chosen correlation function.

    Returns
    -------
    float
        The negative maximum logarithmic negativity between the pairs of the 7 output modes. The negation is used
        since the Brent method finds a minimum, not a maximum.
    """
    # Unpack the args
    h, T, correlation_func, hilbert_space_func = args

    # Compute the Omega and Delta parameters from the effective OmegaR and h
    Omega = 2 * OmegaR * h / (1 + h ** 2)
    Delta = OmegaR * (1 - h ** 2) / (1 + h ** 2)

    # Find the Hilbert space dimensions
    Ns, excitations = hilbert_space_func(T, Omega)

    # Create the time array for the computation
    tlist = np.linspace(0, T, nT)

    # Define the atom with the given parameter set
    driven_atom = DrivenAtom(tlist, Omega, w0=Delta)

    # Perform the correlation function calculation and its decomposition to find the output modes
    vals_list, vecs_list = correlation_func(driven_atom, n=7, trim=False, plot=False)

    # Unpack the vals and vecs for the first dissipation operators
    vals, vecs = vals_list[0], vecs_list[0]

    # Create an array to store the logarithmic negativity
    logarithmic_negativities = np.zeros((len(vecs), len(vecs)))

    # Loop over pairs of output modes
    for n, vec1 in enumerate(vecs):
        for m, vec2 in enumerate(vecs):
            # Only compute the lower triangle of the symmetrix matrix
            if n <= m:
                continue

            # Create an atom with the two output modes and the chosen parameters and Hilbert space
            driven_atom_with_output = DrivenAtomWithOutput([vec1, vec2], tlist, Omega, Delta, Ns, excitations)

            # Evolve the master equation and find the final quantum state
            result: qt.solver.Result = ph.run_quantum_system(driven_atom_with_output, plot=False, verbose='')
            final_state = result.final_state

            # Convert the quantum state from energy restricted dimensions to dense Hilbert space dimensions
            rho_full = convert_enr_to_dense(final_state, Ns, excitations)

            # Compute the logarithmic negativity of the two output modes
            logarithmic_negativities[n, m] = logarithmic_negativity(qt.ptrace(rho_full, [1, 2]))

            # Check that the Hilbert space dimensions were large enough, otherwise state a warning
            expect = result.expect
            if not (np.isclose(np.real(vals[n]), np.real(expect[1][-1]), atol=atol, rtol=rtol)
                    and np.isclose(np.real(vals[m]), np.real(expect[2][-1]), atol=atol, rtol=rtol)):
                Warning(f'The Hilbert space dimensions are not big enough for parameters h = {h}, T = {T}!')

    # Return the negative of the maximum of the logarithmic negativities, since the optimization routines finds minima
    return - np.max(logarithmic_negativities)


class DrivenAtom(QuantumSystem):
    """
    Stores the quantum system for a driven atom. The quantum system is defined as an SLH triple [4] which can be
    evolved in the master equation. The class takes the parameters for the atom and its drive and defines and stores
    the SLH triple.

    Parameters
    ----------
    tlist: np.ndarray
        The time array for the master equation evolution. End point is the parameter T.
    Omega: float
        The pump strength parameter.
    Delta: float
        The pump detuning from the atom transition frequency.
    """

    def __init__(self, tlist: np.ndarray, Omega: float, Delta: float):
        c = qt.destroy(D)
        Id = qt.qeye(D)
        self.c = c
        self.I = Id
        self.gamma = gamma
        self.Omega = Omega
        self.Delta = Delta
        psi0 = qt.basis(D, 0)
        pulses = []
        super().__init__(psi0, pulses, tlist)

    def create_component(self) -> nw.Component:
        """
        Creates the SLH triple which is stored in a component object.

        Returns
        -------
        nw.Component
            The network component which stores the SLH triple.
        """
        driven_atom = nw.Component(S=nw.MatrixOperator(self.I), L=nw.MatrixOperator(np.sqrt(self.gamma) * self.c),
                                   H=self.Omega / 2 * self.c.dag() + np.conjugate(self.Omega) / 2 * self.c
                                     + self.Delta * self.c.dag() @ self.c)
        return driven_atom

    def get_expectation_observables(self) -> Union[List[qt.Qobj], Callable]:
        """
        The observables for which to store the expectation values for.

        Returns
        -------
        list[qt.Qobj]
            List of the observables to compute and store the expectation values for.
        """
        return [self.c1.dag() @ self.c1]

    def get_plotting_options(self) -> Any:
        """
        Plotting options it plot is set to true in the run_quantum_system function.
        """
        pulse_options = []
        content_options = [LineOptions(linetype='-', linewidth=4, color='r',
                                       label=r'$\langle \hat{c}_{1}^\dagger\hat{c}_{1} \rangle$')
                           ]
        return pulse_options, content_options


class DrivenAtomWithOutput(QuantumSystem):
    """
    Stores the quantum system for a driven atom with output. The quantum system is defined as an SLH triple [4] which
    can be evolved in the master equation. The class takes the parameters for the atom and its drive and defines and
    stores the SLH triple. The SLH triple is computed for the serial connection of atom and two output cavities
    using the series product SLH rule [4].

    Parameters
    ----------
    us: List[np.ndarray]
        A list of the output mode arrays.
    tlist: np.ndarray
        The time array for the master equation evolution. End point is the parameter T.
    Omega: float
        The pump strength parameter.
    Delta: float
        The pump detuning from the atom transition frequency.
    Ns: list[int]
        The dimensions of the atom and output mode Hilbert space.
    excitations: int
        The maximum number of excitations for the energy restricted Hilbert space.
    """

    def __init__(self, us: List[np.ndarray], tlist: np.ndarray, Omega: float, Delta: float,
                 Ns: List[int], excitations: int):
        # The number of output modes
        M = len(us)

        # Transform the pulse modes, since their mode shapes are distorted by earlier virtual cavities, see [2]
        vs, gvs = p.transform_pulses(us, tlist, is_input=False)

        # Define the output modes with a Cubic Spline so they can be evaluated at all times
        self.vs = [p.Pulse('undefined', in_going=False,
                           args=[CubicSpline(tlist, vs[i]), CubicSpline(tlist, gvs[i])]) for i in range(M)]

        # Define the energy restricted Hilbert space
        self.c, a1, a2 = qt.enr_destroy(Ns, excitations)
        self.a_list = [a1, a2]
        self.I = qt.enr_identity(Ns, excitations)
        psi0 = qt.enr_fock(Ns, excitations, [0, 0, 0])

        # Store the parameters
        self.gamma = gamma
        self.Omega = Omega
        self.Delta = Delta
        pulses = []
        super().__init__(psi0, pulses, tlist)

    def create_component(self) -> nw.Component:
        """
        Creates the SLH triple which is stored in a component object. The total SLH component is computed by using
        the serial product rule [4], where the two output cavities are placed in series with the atom.

        Returns
        -------
        nw.Component
            The network component which stores the SLH triple.
        """
        # Create a component for the driven atom
        driven_atom = nw.Component(S=nw.MatrixOperator(self.I), L=nw.MatrixOperator(np.sqrt(self.gamma) * self.c),
                                   H=self.Omega / 2 * self.c.dag() + np.conjugate(self.Omega) / 2 * self.c
                                     + self.Delta * self.c.dag() * self.c)
        total_component = driven_atom
        # Add the output mode virtual cavities in series.
        for i, v in enumerate(self.vs):
            cavity = cf.create_cavity(self.I, self.a_list[i], v.g, 0)
            total_component = nw.series_product(total_component, cavity)
        return total_component

    def get_expectation_observables(self) -> List[qt.Qobj]:
        """
        The observables for which to store the expectation values for. We store the expectation values to compare
        with the eigenvalue from the decompositions to make sure the Hilbert space dimensions are large enough.

        Returns
        -------
        list[qt.Qobj]
            List of the observables to compute and store the expectation values for.
        """
        return [self.c.dag() * self.c] + [ai.dag() * ai for ai in self.a_list]

    def get_plotting_options(self) -> Any:
        """
        Plotting options it plot is set to true in the run_quantum_system function.
        """
        pulse_options = []
        adaga_content_options = [LineOptions(linetype='--', linewidth=4, color='r',
                                       label=r'$\langle \hat{a}_{i}^\dagger\hat{a}_{i} \rangle$') for _ in range(len(self.vs))]
        content_options = [LineOptions(linetype='-', linewidth=4, color='r',
                                       label=r'$\langle \hat{c}_{1}^\dagger\hat{c}_{1} \rangle$'),
                           *adaga_content_options]
        return pulse_options, content_options



if __name__ == '__main__':
    main()
