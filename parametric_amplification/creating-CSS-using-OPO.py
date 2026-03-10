"""
creating-CSS-using-OPO.py

Implements the optimization procedure for finding the Coherent State Superposition (CSS) with the largest overlap with
the output squeezed Fock state from an Optical Parametric Oscillator (OPO). The CSS is of the odd type, which is
proportional to |alpha> - |- alpha>. We plot the maps of the alpha for which the overlap between the states are largest,
and we plot a map of the overlaps themselves. The Wigner functions for the squeezed Fock state and the CSS state is
also plotted for a particular parameter set.

The code implements the example given in section VI of {insert reference}.
"""
#! python3
import numpy as np
import qutip as qt
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from squeezing import SqueezingSystem
from scipy.special import factorial
import pickle
from opo import OpticalParametricOscillator


M = 100  # the Hilbert space size of the output numerical quantum state

file = r'cat_sq_fock_overlap.pkl'  # The file for saving the result
overwrite = False


def main():
    # If the file does not exist or overwrite is enabled, calculate the maps and save them to the file
    if not os.path.exists(file) or overwrite:
        optimize_cat()
    # Plot the alpha and overlap maps
    plot_alpha_overlaps()
    # Plot the Wigner functions for a specific parameter set
    plot_wigner_functions()


def plot_alpha_overlaps():
    """
    Plots the map of the alpha which gives the best state overlap and the overlaps themselves.
    """
    # Opens the save file and loads the results
    with open(file, 'rb') as f:
        dic = pickle.load(f)
        rs = dic['rs']
        Gammas = dic['Gammas']
        max_alphas = dic['max_alphas']
        max_overlaps = dic['max_overlaps']

    # Plot the alpha map
    plt.pcolormesh(rs, Gammas, np.real(max_alphas), vmin=0, vmax=3)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\Gamma$')
    plt.xticks([0.1, 0.5, 1, 1.5, 2, 2.5])
    plt.yticks([0.25, 5, 10, 15])
    plt.title(rf'Max $\alpha$ sq. fock and odd cat state')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('cat_sq_fock_overlap_alphas.png')
    plt.show()

    # Plot the overlap map
    plt.pcolormesh(rs, Gammas, np.real(max_overlaps), vmin=0, vmax=1)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\Gamma$')
    plt.xticks([0.1, 0.5, 1, 1.5, 2, 2.5])
    plt.yticks([0.25, 5, 10, 15])
    plt.title(f'Max overlap sq. fock and odd cat state')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('cat_sq_fock_overlap_overlaps.png')
    plt.show()


def plot_wigner_functions():
    """
    Plots the Wigner functions for the squeezed Fock state and the odd cat state for a specific set of parameters.
    """
    # Opens the save file and loads the results
    with open(file, 'rb') as f:
        dic = pickle.load(f)
        Gammas = dic['Gammas']
        rs = dic['rs']
        alphas = dic['max_alphas']

    # Choose a specific set of parameters
    Gamma = Gammas[0]
    r = rs[64]
    alpha = alphas[0, 64]

    # Find the squeezed Fock state for these parameters
    sq_fock = qt.ptrace(squeezed_fock_state(xi=r, Gamma=Gamma), 0)

    # Find the odd cat state for these parameters
    sq_cat = cat_state(alpha)

    # Find the Wigner function numerically of the states
    xvec = np.linspace(-5, 5, 200)
    w_fock = qt.wigner(sq_fock, xvec, xvec)
    w_cat = qt.wigner(sq_cat, xvec, xvec)

    # Find the range of values for the plot
    w_max = max(w_fock.max(), w_cat.max())
    w_min = min(w_fock.min(), w_cat.min())
    w_abs = max(abs(w_max), abs(w_min))

    # Normalize the colours in the plot, such that 0 is white
    nrm = mpl.colors.Normalize(w_min * w_abs / abs(w_min), w_max * w_abs / abs(w_max))

    # Plot the Wigner functions
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    axs[0].contourf(xvec, xvec, w_fock, 100, cmap=mpl.cm.RdBu, norm=nrm)
    axs[1].contourf(xvec, xvec, w_cat, 100, cmap=mpl.cm.RdBu, norm=nrm)
    for i in range(2):
        axs[i].set_aspect('equal')

    sm = mpl.cm.ScalarMappable(norm=nrm, cmap=mpl.cm.RdBu)
    sm.set_array([])  # required – the array itself isn’t used

    cbar = fig.colorbar(
        sm,
        ax=axs,  # attach to all contour axes
        location='right',
        shrink=1,  # make it a bit shorter than the full height
        pad=0.02,  # distance from the figure edge
    )
    plt.savefig('squeezed_fock_and_cat.png')
    plt.show()

    # Find the overlap and print it
    res = sq_cat.overlap(sq_fock)
    print(f'overlap for Gamma = {Gamma}, r = {r} and alpha = {alpha} is = {res * np.conjugate(res)}')


def optimize_cat():
    """
    Runs the optimization algorithm, which find the odd cat state with the greatest overlap for each set of parameters
    for the input pulse and OPO which will generate the squeezd Fock state.
    """
    # Define a grid of alphas to attempt optimizing over
    alphas = np.linspace(0.1, 6, 200)

    # Define an array for the cat states for each alpha
    cats_array = np.zeros((len(alphas), M, M), dtype=np.complex128)

    # Fill the grid with odd cat states for each value of alpha
    for i, alpha in enumerate(alphas):
        # Find the odd cat state for this alpha value
        cat = cat_state(alpha)
        # Get the numpy array
        sq_cat_array = cat.data_as('ndarray')
        # Save it in the array for all cats. Save the density matrix instead of the state vector
        cats_array[i, :, :] = np.outer(sq_cat_array.conjugate(), sq_cat_array)

    # Define parameters for the OPO strength and pulse length
    rs = np.linspace(0.1, 2.5, 100)
    Gammas = np.linspace(0.25, 15, 100)

    # Create an array of values for parallelization routine
    values = []
    for Gamma in Gammas:
        for r in rs:
            values.append((r, Gamma))

    # Create an array for the squeezed fock states
    sq_focks_array = np.zeros((len(Gammas), len(rs), M, M), dtype=np.complex128)
    shape = sq_focks_array.shape

    # Calculate the output squeezed fock in a parallel routine
    sq_fock_list = qt.parallel_map(create_squeezed_fock_state, values, progress_bar='enhanced')
    sq_focks_array = np.array(sq_fock_list).reshape(shape)

    # Perform the quantum state overlap between the squeezed Fock state and the odd cat states
    overlaps = np.einsum('ijnm, kmn -> ijk', sq_focks_array, cats_array)

    # Compute the norm square of the overlaps
    overlaps = overlaps * np.conjugate(overlaps)

    # Find the maximum overlaps for each r, Gamma value, and record the alpha that results in this maximum
    max_idx = np.argmax(overlaps, axis=2)
    max_overlaps = np.max(overlaps, axis=2)
    max_alphas = alphas[max_idx]

    # Save the result to the file
    with open(file, 'wb') as f:
        pickle.dump({'rs': rs, 'Gammas': Gammas, 'max_alphas': max_alphas, 'max_overlaps': max_overlaps}, f)


def create_squeezed_fock_state(val) -> np.ndarray:
    """
    Creates the output squeezed Fock state from the OPO using the method in the referenced paper.

    Parameters
    ----------
    val : tuple(float, float)
        The squeezing strength r, and pulse width Gamma pair

    Returns
    -------
    ndarray
        The numpy array representing the density matrix of the output squeezed Fock state
    """
    # Unpack the r, Gamma pair
    r, Gamma = val
    try:
        # Run the squeezed Fock state routine, trace over the second mode, and return the data as a numpy array
        return qt.ptrace(squeezed_fock_state(xi=r, Gamma=Gamma), 0).data_as('ndarray')
    except Exception:
        # If the routine fails due to the discontinuity in the OPO input-output relation, return an array of zeros
        return np.zeros((M, M))


def squeezed_fock_state(xi: complex = None, Gamma: float = None) -> qt.Qobj:
    """
    Calculates the output squeezed Fock state of an OPO irradiation by an exponential pulse shape with decay rate Gamma.

    Parameters
    ----------
    xi: complex
        The driving strength of the OPO.
    Gamma: float
        The decay rate of the input pulse shape, u(t) = sqrt(Gamma) exp(-Gamma t / 2).

    Returns
    -------
    Qobj
        The quantum object that represents the two-mode output quantum state of the OPO.
    """
    # Physical parameters
    if Gamma is None:
        Gamma = 0.25
    gamma = 1           # decay rate of the open quantum system
    if xi is None:
        xi = 0.22           # amplitude strength of the parametric oscillator
    n     = 1           # number of photons in the fock state
    Delta = 0           # detuning between the parametric oscillators and the input pulse

    # Array of frequencies for the relevant spectrum
    omegas = np.linspace(-300, 300, 30000)

    # A one-sided exponential input pulse in frequency domain
    u = lambda omega: np.sqrt(Gamma / (2 * np.pi)) * (1j * omega + Gamma / 2) / (omega ** 2 + Gamma ** 2 / 4)

    # Create an OPO with the given parameters
    opo = OpticalParametricOscillator(gamma, xi, Delta)

    # Transform the input pulse by F and G
    zeta_u, xi_u, fu, gu = opo.get_fu_and_gu(omegas, u)

    # Creation operator for a fock state
    def f(au: qt.Qobj, audag: qt.Qobj) -> qt.Qobj:
        """
        Creates a Fock state in a given mode under the action upon the vacuum state

        Parameters
        ----------
        au : qt.Qobj
            The annihilation operator of the mode
        audag : qt.Qobj
            The creation operator of the mode

        Returns
        -------
        qt.Qobj
            The cat state operator
        """
        return audag ** n / np.sqrt(factorial(n))

    # Initialize the squeezing system with the system parameters
    ss = SqueezingSystem(f, M, u, fu, gu, zeta_u, xi_u, omegas)

    # Compute the output modes (two modes when the input is in a fock state)
    v1, v2 = ss.get_output_modes()

    # Transform the output pulses by F and G
    zeta_v1, xi_v1, zeta_v2, xi_v2, fv1, gv1, fv2, gv2 = opo.get_fv_and_gv(omegas, [v1, v2])

    # Obtain the full density matrix of the squeezed fock state
    rho_squeezed_fock_state = ss.get_squeezed_output_state([v1, v2], [fv1, fv2], [gv1, gv2],
                                                           [zeta_v1, zeta_v2], [xi_v1, xi_v2])

    return rho_squeezed_fock_state


def cat_state(alpha: complex) -> qt.Qobj:
    """
    Generates an odd cat state of form |cat> = |alpha> - |- alpha>.

    Parameters
    ----------
    alpha : complex
        The amplitude of the coherent state used to generate the cat state

    Returns
    -------
    Qobj
        A quantum object representing the odd cat state
    """
    # Define the displacement operators for the coherent state superpositions
    Dp = qt.displace(M, alpha)
    Dm = qt.displace(M, -alpha)

    # Define a vacuum state to act upon
    psi0 = qt.basis(M, 0)

    # Create the odd cat state through multiplication and subsequent normalization
    cat = ((Dp - Dm) @ psi0).unit()
    return cat


if __name__ == '__main__':
    main()
