#! python3
import numpy as np
import qutip as qt
import matplotlib as mpl
import matplotlib.pyplot as plt
from qutip import ket2dm

from squeezing import overlap, SqueezingSystem
from scipy.special import factorial
from cmath import phase
import pickle
import random
from opo import OpticalParametricOscillator


M = 100  # the Hilbert space size of the output numerical quantum state


def main():
    #plot_squeezed_cat_and_sq_fock_state()
    #optimize_squeezed_cat()
    optimize_cat()
    return
    rho_exp = experimental_state()
    print(f'Norm of exp. state is {rho_exp.norm()}')

    xi_primes = np.linspace(0, 1, 100)
    alphas = np.linspace(2, 3, 20)

    xi_max = 0
    alpha_max = 0
    max_val = 0
    for xi_prime in xi_primes:
        for alpha in alphas:
            rho_ideal = squeezed_cat_state2(xi_prime, alpha)
            val = rho_exp.overlap(rho_ideal)
            if val * np.conjugate(val) > max_val * np.conjugate(max_val):
                xi_max = xi_prime
                alpha_max = alpha
                max_val = val

    rho_ideal = squeezed_cat_state2(xi_max, alpha_max, plot=True)

    print(f'Max overlap with cat state of {max_val} at xi = {xi_max},',
          f'|alpha|^2 = {alpha_max * np.conjugate(alpha_max)}')


def plot_squeezed_cat_and_sq_fock_state():
    r1 = 0
    r2 = 0.76 * np.exp(1j * np.pi)
    alpha = 1.8

    sq_fock = qt.squeeze(M, r2) @ qt.create(M) @ qt.basis(M, 0)
    sq_cat = qt.squeeze(M, r1) @ ((qt.displace(M, alpha) - qt.displace(M, -alpha)) @ qt.basis(M, 0)).unit()

    # Find the Wigner function numerically of output mode 1
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(sq_fock, xvec, xvec)

    # Plot the Wigner function for output mode 1
    nrm = mpl.colors.Normalize(w.min(), w.max())
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    # Find the Wigner function numerically of output mode 2
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(sq_cat, xvec, xvec)

    # Plot the Wigner function for output mode 2
    nrm = mpl.colors.Normalize(w.min(), w.max())
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    res = sq_cat.overlap(sq_fock)
    print(res * np.conjugate(res))


def optimize_cat():
    #r2s = np.linspace(0, 2, 50)
    alphas = np.linspace(0.1, 6, 200)

    cats_array = np.zeros((len(alphas), M, M), dtype=np.complex128)
    #sq_cats_array = np.zeros((len(r2s), len(alphas), M, M), dtype=np.complex128)

    #for i, r2 in enumerate(r2s):
    #    S = qt.squeeze(M, r2)
    I = qt.qeye(M)
    for i, alpha in enumerate(alphas):
        sq_cat = squeezed_cat_state(I, alpha)
        sq_cat_array = sq_cat.data_as('ndarray')
        cats_array[i, :, :] = np.outer(sq_cat_array.conjugate(), sq_cat_array)

    rs = np.linspace(0.1, 2.5, 100)
    Gammas = np.linspace(0.25, 15, 100)

    values = []

    for Gamma in Gammas:
        for r in rs:
            values.append((r, Gamma))

    sq_focks_array = np.zeros((len(Gammas), len(rs), M, M), dtype=np.complex128)
    shape = sq_focks_array.shape

    sq_fock_list = qt.parallel_map(create_squeezed_fock_state, values, progress_bar='enhanced')
    sq_focks_array = np.array(sq_fock_list).reshape(shape)

    overlaps = np.einsum('ijnm, kmn -> ijk', sq_focks_array, cats_array)

    overlaps = overlaps * np.conjugate(overlaps)
    max_idx = np.argmax(overlaps, axis=2)
    max_overlaps = np.max(overlaps, axis=2)
    #flat = overlaps.reshape(overlaps.shape[:2] + (-1,))
    #flat_argmax = np.argmax(flat, axis=2)
    #max_idx_x, max_idx_y = np.unravel_index(flat_argmax, (overlaps.shape[2], overlaps.shape[3]))
    #max_overlaps = np.max(flat, axis=2)

    #max_r2s = r2s[max_idx_x]
    #max_alphas = alphas[max_idx_y]
    max_alphas = alphas[max_idx]

    file = r'cat_sq_fock_overlap.pkl'
    with open(file, 'wb') as f:
        pickle.dump({'rs': rs, 'Gammas': Gammas, 'max_alphas': max_alphas, 'max_overlaps': max_overlaps}, f)

    #plt.pcolormesh(rs, Gammas, max_r2s)
    #plt.xlabel(r'$r$')
    #plt.ylabel(r'$\Gamma$')
    #plt.title(rf'Max $r$ sq. fock and sq. odd cat state')
    #plt.legend()
    #plt.colorbar()
    #plt.tight_layout()
    #plt.show()

    plt.pcolormesh(rs, Gammas, max_alphas)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\Gamma$')
    plt.title(rf'Max $\alpha$ sq. fock and sq. odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.pcolormesh(rs, Gammas, np.real(max_overlaps))
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\Gamma$')
    plt.title(f'Max overlap sq. fock and sq. odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def create_squeezed_fock_state(val):
    r, Gamma = val
    try:
        return qt.ptrace(squeezed_fock_state(xi=r, Gamma=Gamma), 0).data_as('ndarray')
    except Exception:
        return np.zeros((M, M))


def optimize_squeezed_cat():
    sq_fock = qt.ptrace(squeezed_fock_state(), 0)
    print(qt.expect(qt.qeye(M), sq_fock))

    rs = np.linspace(0, 1, 50)
    phis = np.linspace(0, np.pi, 25)
    alphas = np.linspace(0.1, 6, 50)

    max_rs = np.zeros((len(rs), len(phis)))
    max_phis = np.zeros((len(rs), len(phis)))
    max_alphas = np.zeros((len(rs), len(phis)))
    max_overlaps = np.zeros((len(rs), len(phis)))
    for i, r in enumerate(rs):
        print(f'i = {i}')
        for j, phi in enumerate(phis):
            S = qt.squeeze(M, r * np.exp(1j * phi))
            for k, alpha in enumerate(alphas):
                sq_cat = squeezed_cat_state(S, alpha)
                res = qt.expect(sq_fock, sq_cat)
                if max_overlaps[i, j] < res * np.conjugate(res):
                    max_overlaps[i, j] = res * np.conjugate(res)
                    max_alphas[i, j] = alpha
                    max_phis[i, j] = phi
                    max_rs[i, j] = r

    plt.pcolormesh(rs, phis, max_rs.T)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\phi$')
    plt.title(rf'Max $r$ sq. fock and odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.pcolormesh(rs, phis, max_phis.T)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\phi$')
    plt.title(rf'Max $\phi$ sq. fock and odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.pcolormesh(rs, phis, max_alphas.T)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\phi$')
    plt.title(rf'Max $\alpha$ sq. fock and odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.pcolormesh(rs, phis, max_overlaps.T)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\phi$')
    plt.title(f'Max overlap sq. fock and odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    print(np.max(max_overlaps))
    print(max_alphas[0, 0])
    print(max_overlaps[0, 0])



def squeezed_fock_state(xi=None, Gamma=None):
    """Generate and plot the Wigner function of a squeezed fock state. In this example we follow the full procedure
    of the original work, and show how the code package functionality can be used for each step. This generates the
    example of a squeezed fock state in the original work."""
    # Physical parameters
    if Gamma is None:
        Gamma = 0.25
    gamma = 1           # decay rate of the open quantum system
    if xi is None:
        xi = 0.22           # amplitude strength of the parametric oscillator
    n     = 1           # number of photons in the fock state
    #tp    = 0           # pulse center in time of input gaussian pulse
    #tau   = 10           # pulse width in time of input gaussian pulse
    Delta = 0           # detuning between the parametric oscillators and the input pulse

    # Array of frequencies for the relevant spectrum
    omegas = np.linspace(-300, 300, 30000)

    # A gaussian input pulse in frequency domain (fourier transform of time domain)
    #u = lambda omega: np.sqrt(tau) / np.pi ** (1 / 4) * np.exp(-tau ** 2 / 2 * omega ** 2 + 1j * tp * omega)
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
    # Find the Wigner function numerically of output mode 1
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(qt.ptrace(rho_squeezed_fock_state, 0), xvec, xvec)

    # Plot the Wigner function for output mode 1
    nrm = mpl.colors.Normalize(w.min(), w.max())
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    # Find the Wigner function numerically of output mode 2
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(qt.ptrace(rho_squeezed_fock_state, 1), xvec, xvec)

    # Plot the Wigner function for output mode 2
    nrm = mpl.colors.Normalize(w.min(), w.max())
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    return rho_squeezed_fock_state


def squeezed_cat_state(S: qt.Qobj, alpha):
    Dp = qt.displace(M, alpha)
    Dm = qt.displace(M, -alpha)
    psi0 = qt.basis(M, 0)
    sq_cat = S @ ((Dp - Dm) @ psi0).unit()
    return sq_cat


def experimental_state():
    """Generate and plot the Wigner function of a squeezed fock state. In this example we follow the full procedure
    of the original work, and show how the code package functionality can be used for each step. This generates the
    example of a squeezed fock state in the original work."""
    # Physical parameters
    gamma   = 1           # decay rate of the open quantum system
    xi      = 0.1j         # amplitude strength of the parametric oscillator
    n       = 3           # number of heralded photons
    tp      = 4           # pulse center in time of input gaussian pulse
    tau     = 1           # pulse width in time of input gaussian pulse
    Delta   = 0           # detuning between the parametric oscillators and the input pulse
    M       = 20          # the Hilbert space size of the output numerical quantum state
    epsilon = 0.2 * np.tanh(xi)        # Mixing angle of half-wave plate

    # Array of frequencies for the relevant spectrum
    omegas = np.linspace(-6, 6, 1000)

    # A gaussian input pulse in frequency domain (fourier transform of time domain)
    u = lambda omega: np.sqrt(tau) / np.pi ** (1 / 4) * np.exp(-tau ** 2 / 2 * omega ** 2 + 1j * tp * omega)

    # Create an OPO with the given parameters
    opo = OpticalParametricOscillator(gamma, xi, Delta)

    # Transform the input pulse by F and G
    zeta_u, xi_u, fu, gu = opo.get_fu_and_gu(omegas, u)

    # Creation operator for a fock state
    def f(au: qt.Qobj, audag: qt.Qobj) -> qt.Qobj:
        """
        Creates a mix of two squeezed state that has been heralded with 2 photons as in PRL 115, 023602 (2015)
        in a given mode under the action upon the vacuum state

        Parameters
        ----------
        au : qt.Qobj
            The annihilation operator of the mode
        audag : qt.Qobj
            The creation operator of the mode

        Returns
        -------
        qt.Qobj
            The state creation operator
        """
        return ((epsilon * np.sqrt(n * (n - 1)) * au * (audag) ** (n - 1) / np.sqrt(factorial(n - 1) * (n - 1))
                 + np.tanh(xi) * audag ** n / np.sqrt(factorial(n)))
                / np.sqrt(n * (n - 1) * epsilon ** 2 + np.tanh(xi) ** 2))

    # Initialize the squeezing system with the system parameters
    ss = SqueezingSystem(f, M, u, fu, gu, zeta_u, xi_u, omegas)

    # Compute the output modes (two modes when the input is in a fock state)
    v1, v2 = ss.get_output_modes()

    # Transform the output pulses by F and G
    zeta_v1, xi_v1, zeta_v2, xi_v2, fv1, gv1, fv2, gv2 = opo.get_fv_and_gv(omegas, [v1, v2])

    au = qt.destroy(M)
    psi0 = qt.basis(M, 0)

    rho = qt.ket2dm(f(au, au.dag()) @ psi0)

    # Find the Wigner function numerically of output mode 1
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(rho, xvec, xvec)

    # Plot the Wigner function for output mode 1
    nrm = mpl.colors.Normalize(w.min(), w.max())
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    # Obtain the full density matrix of the squeezed fock state
    rho_squeezed_state = ss.get_squeezed_output_state([v1, v2], [fv1, fv2], [gv1, gv2],
                                                           [zeta_v1, zeta_v2], [xi_v1, xi_v2])

    # Find the Wigner function numerically of output mode 1
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(qt.ptrace(rho_squeezed_state, 0), xvec, xvec)

    # Plot the Wigner function for output mode 1
    nrm = mpl.colors.Normalize(w.min(), w.max())
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    # Find the Wigner function numerically of output mode 2
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(qt.ptrace(rho_squeezed_state, 1), xvec, xvec)

    # Plot the Wigner function for output mode 2
    nrm = mpl.colors.Normalize(w.min(), w.max())
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    return qt.ptrace(rho_squeezed_state, 0)


def squeezed_cat_state2(xi_prime, alpha, plot=False):
    cat = ((qt.coherent(20, alpha) - qt.coherent(20, - alpha))
           / np.sqrt(2 * (1 - np.exp(-2 * alpha * np.conjugate(alpha)))))

    squeezed_cat = qt.squeeze(20, xi_prime) @ cat
    norm = squeezed_cat.norm()

    if plot:
        # Find the Wigner function numerically of output mode 1
        xvec = np.linspace(-5, 5, 200)
        w = qt.wigner(qt.ket2dm(squeezed_cat), xvec, xvec)

        # Plot the Wigner function for output mode 1
        nrm = mpl.colors.Normalize(w.min(), w.max())
        fig, axs = plt.subplots(1, 1)
        cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
        plt.colorbar(cs)
        plt.show()

    return squeezed_cat / norm


def ideal_single_mode_case_cat_state():
    """Computes the overlap between a pi/2 squeezed schrödinger cat state and a photon added squeezed vacuum state
    both in a single ideal mode, to benchmark the potential ideal operation case"""
    N = 400

    phi = np.pi

    rs = np.linspace(0.1, 4, 1000)
    alphas = np.linspace(0.1, 20, 1000)

    overlaps = np.zeros((len(rs), len(alphas)))
    for i, r in enumerate(rs):
        print('i = ', i)
        for j, alpha in enumerate(alphas):
            current_overlap = analytical_cat_added_squeezed_state_overlap(r, phi, alpha, N)
            overlaps[i, j] = current_overlap * np.conjugate(current_overlap)

    plt.pcolormesh(rs, alphas, overlaps)
    plt.xlabel('$r$')
    plt.ylabel(r'$\alpha$')
    plt.title(f'Overlap photon added sq. vac and odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def ideal_single_mode_case_cat_state_squeezed_fock_state():
    """Computes the overlap between a pi/2 squeezed schrödinger cat state and a squeezed fock state
    both in a single ideal mode, to benchmark the potential ideal operation case"""
    N = 400

    rs = np.linspace(0.1, 2, 50)
    phis = np.linspace(0, 2*np.pi, 50)
    alphas = np.linspace(0.1, 6, 50) + 1j * np.linspace(0.1, 6, 50)

    max_alphas = np.zeros((len(rs), len(phis)))
    max_overlaps = np.zeros((len(rs), len(phis)))
    for i, r in enumerate(rs):
        print('i = ', i)
        for j, phi in enumerate(phis):
            for k, alpha in enumerate(alphas):
                current_overlap = analytical_cat_state_squeezed_fock_state_overlap(r, phi, alpha, N)
                if current_overlap * np.conjugate(current_overlap) > max_overlaps[i, j]:
                    max_overlaps[i, j] = current_overlap * np.conjugate(current_overlap)
                    max_alphas[i, j] = np.sqrt(alpha * np.conjugate(alpha))

    plt.pcolormesh(phis, rs, max_alphas)
    plt.xlabel(r'$\phi$')
    plt.ylabel('$r$')
    plt.title(f'Max alpha sq. fock and odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.pcolormesh(phis, rs, max_overlaps)
    plt.xlabel(r'$\phi$')
    plt.ylabel('$r$')
    plt.title(f'Max overlap sq. fock and odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def ideal_single_mode_case_squeezed_cat_state():
    """Computes the overlap between a pi/2 squeezed schrödinger cat state and a photon added squeezed vacuum state
    both in a single ideal mode, to benchmark the potential ideal operation case"""
    N = 50
    psi0 = qt.basis(N, 0)
    a = qt.destroy(N)

    rs = np.linspace(0.1, 1, 10)
    phis = np.linspace(0, 2*np.pi, 10)
    alphas = np.linspace(0.1, 1, 10)

    for i, r1 in enumerate(rs):
        print('i = ', i)
        for j, phi1 in enumerate(phis):
            S1 = qt.squeeze(N, r1 * np.exp(1j * phi1))
            for k, r2 in enumerate(rs):
                for n, phi2 in enumerate(phis):
                    S2 = qt.squeeze(N, r2 * np.exp(1j * phi2))
                    for m, alpha in enumerate(alphas):
                        Dp = qt.displace(N, alpha)
                        Dm = qt.displace(N, -alpha)
                        cat = ((Dp - Dm) @ psi0).unit()
                        numerical_overlap = cat.overlap((S1.dag() @ a.dag() @ S2 @ psi0).unit())
                        print(analytical_squeezed_cat_added_squeezed_state_overlap(r1, phi1, alpha, r2, phi2, N) - numerical_overlap)


def ideal_cat_state_doubly_squeezed_fock_state_overlap():
    N = 150

    rs = np.linspace(0.1, 4, 30)
    phis = np.linspace(0, 2*np.pi, 30)
    alphas = np.linspace(0.1, 10, 30)

    max_phi1s = np.zeros((len(rs), len(rs)))
    max_phi2s = np.zeros((len(rs), len(rs)))
    max_alphas = np.zeros((len(rs), len(rs)))
    max_overlaps = np.zeros((len(rs), len(rs)))
    for i, r1 in enumerate(rs):
        print(f'i = {i}')
        for j, r2 in enumerate(rs):
            for k, phi1 in enumerate(phis):
                for m, phi2 in enumerate(phis):
                    for n, alpha in enumerate(alphas):
                        res = cat_state_doubly_squeezed_fock_state_overlap(r1, phi1, r2, phi2, alpha, N)
                        if max_overlaps[i, j] < res * np.conjugate(res):
                            max_overlaps[i, j] = res * np.conjugate(res)
                            max_phi1s[i, j] = phi1
                            max_phi2s[i, j] = phi2
                            max_alphas[i, j] = alpha

    plt.pcolormesh(rs, rs, max_phi1s.T)
    plt.xlabel(r'$r$-prime')
    plt.ylabel(r'$r$')
    plt.title(rf'Max $\phi_1$ sq. fock and odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.pcolormesh(rs, rs, max_phi2s.T)
    plt.xlabel(r'$r$-prime')
    plt.ylabel(r'$r$')
    plt.title(rf'Max $\phi_2$ sq. fock and odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.pcolormesh(rs, rs, max_alphas.T)
    plt.xlabel(r'$r$-prime')
    plt.ylabel(r'$r$')
    plt.title(rf'Max $\alpha$ sq. fock and odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.pcolormesh(rs, rs, max_overlaps.T)
    plt.xlabel(r'$r$-prime')
    plt.ylabel(r'$r$')
    plt.title(f'Max overlap sq. fock and odd cat state')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def analytical_photon_added_squeezed_state(r, phi, N):
    state_array = np.zeros(N, dtype=np.complex128)
    for n in range(N // 2):
        state_array[2 * n + 1] = (- np.exp(1j * phi) * np.tanh(r)) ** n * np.sqrt(factorial(2*n + 1)) / (2 ** n * factorial(n))
    state_array /= np.cosh(r) ** (3 / 2)
    return qt.Qobj(state_array, dims=[N, 1])


def analytical_odd_cat(alpha, N):
    factor = 2 * np.exp(- alpha * np.conjugate(alpha) / 2) / np.sqrt(2 * (1 - np.exp(- 2 * alpha * np.conjugate(alpha))))
    state_array = np.zeros(N, dtype=np.complex128)
    for n in range(N // 2):
        state_array[2 * n + 1] = alpha ** (2 * n + 1) / np.sqrt(factorial(2 * n + 1))
    state_array *= factor
    return qt.Qobj(state_array, dims=[N, 1])


def analytical_cat_added_squeezed_state_overlap(r, phi, alpha, N):
    factor = (2 * np.exp(- alpha * np.conjugate(alpha) / 2)
              / (np.cosh(r) ** (3 / 2) * np.sqrt(2 * (1 - np.exp(- 2 * alpha * np.conjugate(alpha))))))
    terms = np.arange(N)
    def f(n):
        logs = - np.log(factorial(n)) + n * np.log(np.tanh(r)) + (2 * n + 1) * np.log(alpha) - n * np.log(2)
        return (- np.exp(- 1j * phi)) ** n * np.exp(logs)
        #return (- np.exp(- 1j * phi) * np.tanh(r)) ** n * alpha ** (2 * n + 1) / (2 ** n * factorial(n))
    return np.sum(f(terms)) * factor


def analytical_squeezed_cat_added_squeezed_state_overlap(r1, phi1, alpha, r2, phi2, N):
    t1 = - np.exp(1j * phi1) * np.tanh(r1)
    t2 = np.exp(1j * phi2) * np.tanh(r2)
    t3 = (t1 + t2) / (1 + np.conjugate(t1) * t2)
    r3 = np.atanh(np.sqrt(t3 * np.conjugate(t3)))
    exp_factor = np.exp(np.log((1 + t1 * np.conjugate(t2)) / (1 + np.conjugate(t1) * t2)) / 4)
    factor = 2 * exp_factor * np.exp(- alpha * np.conjugate(alpha) / 2) / np.sqrt(np.cosh(r3) * 2 * (1 - np.exp(- 2 * alpha * np.conjugate(alpha)))) / np.cosh(r2)

    terms1 = np.arange(N)
    def f1(n):
        logs = - np.log(factorial(n)) + (2 * n + 1) * np.log(alpha) - n * np.log(2)
        return (- t3) ** n * np.exp(logs)

    terms2 = np.arange(1, N)
    def f2(n):
        logs = - np.log(factorial(n)) + (2 * n - 1) * np.log(alpha) - n * np.log(2)
        return 2 * n * (- t3) ** n * np.exp(logs)

    res = factor * (np.cosh(r1) * np.sum(f1(terms1)) + np.sinh(r1) * np.exp(-1j * phi1) * np.sum(f2(terms2)))
    return res


def analytical_squeezed_fock_state(r, phi, N):
    state_array = np.zeros(N, dtype=np.complex128)
    for n in range(N // 2):
        logs = - np.log(factorial(n)) + n * np.log(np.tanh(r)) - n * np.log(2) + 0.5 * np.log(factorial(2 * n + 1))
        state_array[2 * n + 1] += np.sqrt(np.cosh(r)) * (- np.exp(1j * phi)) ** n * np.exp(logs)

    for n in range(1, N // 2):
        logs = - np.log(factorial(n)) + n * np.log(np.tanh(r)) - n * np.log(2) + 0.5 * np.log(factorial(2 * n - 1))
        state_array[2 * n - 1] += np.sinh(r) * np.exp(-1j * phi) / np.sqrt(np.cosh(r)) * 2 * n * (- np.exp(1j * phi)) ** n * np.exp(logs)

    return qt.Qobj(state_array, dims=[N, 1])


def analytical_cat_state_squeezed_fock_state_overlap(r, phi, alpha, N):
    factor = np.sqrt(2 * np.exp(-alpha * np.conjugate(alpha)) / (1 - np.exp(-2 * alpha * np.conjugate(alpha))))

    terms1 = np.arange(N)
    def f1(n):
        logs = - np.log(factorial(n)) + n * np.log(np.tanh(r)) - n * np.log(2) + (2*n + 1) * np.log(alpha)
        return (- np.exp(1j * phi)) ** n * np.exp(logs)

    terms2 = np.arange(1, N)
    def f2(n):
        logs = - np.log(factorial(n)) + n * np.log(np.tanh(r)) - n * np.log(2) + (2*n - 1) * np.log(alpha)
        return (- np.exp(1j * phi)) ** n * 2 * n * np.exp(logs)

    return factor * (np.sqrt(np.cosh(r)) * np.sum(f1(terms1))
            + np.sinh(r) * np.exp(-1j * phi) / np.sqrt(np.cosh(r)) * np.sum(f2(terms2)))


def doubly_squeezed_state(r1, phi1, r2, phi2, N):
    psi0 = qt.basis(N, 0)
    S1 = qt.squeeze(N, - r1 * np.exp(1j * phi1))
    S2 = qt.squeeze(N, r2 * np.exp(1j * phi2))

    psi1 = S1 @ S2 @ psi0

    t1 = - np.exp(1j * phi1) * np.tanh(r1)
    t2 = np.exp(1j * phi2) * np.tanh(r2)
    t3 = (t1 + t2) / (1 + np.conjugate(t1) * t2)
    r3 = np.atanh(np.sqrt(t3 * np.conjugate(t3)))
    phi3 = phase(t3)

    S3 = qt.squeeze(N, r3 * np.exp(1j * phi3))

    psi2 = ((1 + t1 * np.conjugate(t2)) / (1 + np.conjugate(t1) * t2))**(1/4) * S3 @ psi0

    print(psi1.overlap(psi2))


def doubly_squeezed_fock_state(r1, phi1, r2, phi2, N):
    t1 = - np.exp(1j * phi1) * np.tanh(r1)
    t2 = np.exp(1j * phi2) * np.tanh(r2)
    t3 = (t1 + t2) / (1 + np.conjugate(t1) * t2)
    r3 = np.atanh(np.sqrt(t3 * np.conjugate(t3)))

    factor = ((1 + t1 * np.conjugate(t2)) / (1 + np.conjugate(t1) * t2))**(1/4)
    factor1 = factor * (np.cosh(r2) * np.cosh(r1) - np.sinh(r2) * np.sinh(r1) * np.exp(1j * (phi1 - phi2)))
    factor2 = factor * (np.sinh(r2) * np.cosh(r1) * np.exp(-1j * phi2) - np.cosh(r2) * np.sinh(r1) * np.exp(-1j * phi1))

    state_array = np.zeros(N, dtype=np.complex128)
    for n in range(N // 2):
        logs = - np.log(factorial(n)) - n * np.log(2) + 0.5 * np.log(factorial(2 * n + 1))
        state_array[2 * n + 1] += factor1 * (- t3) ** n * np.exp(logs) / np.sqrt(np.cosh(r3))

    for n in range(1, N // 2):
        logs = - np.log(factorial(n)) - n * np.log(2) + 0.5 * np.log(factorial(2 * n - 1))
        state_array[2 * n - 1] += factor2 * 2 * n * (- t3) ** n * np.exp(logs) / np.sqrt(np.cosh(r3))

    return qt.Qobj(state_array, dims=[N, 1])


def cat_state_doubly_squeezed_fock_state_overlap(r1, phi1, r2, phi2, alpha, N):
    t1 = - np.exp(1j * phi1) * np.tanh(r1)
    t2 = np.exp(1j * phi2) * np.tanh(r2)
    t3 = (t1 + t2) / (1 + np.conjugate(t1) * t2)
    r3 = np.atanh(np.sqrt(t3 * np.conjugate(t3)))

    factor = ((1 + t1 * np.conjugate(t2)) / (1 + np.conjugate(t1) * t2))**(1/4)
    cat_factor = np.sqrt(2 * np.exp(- alpha * np.conjugate(alpha)) / (1 - np.exp(- 2 * alpha * np.conjugate(alpha))))
    factor1 = factor * cat_factor * (np.cosh(r2) * np.cosh(r1) - np.sinh(r2) * np.sinh(r1) * np.exp(1j * (phi1 - phi2))) / np.sqrt(np.cosh(r3))
    factor2 = factor * cat_factor * (np.sinh(r2) * np.cosh(r1) * np.exp(-1j * phi2) - np.cosh(r2) * np.sinh(r1) * np.exp(-1j * phi1)) / np.sqrt(np.cosh(r3))

    terms1 = np.arange(N)
    def f1(n):
        logs = - np.log(factorial(n)) - n * np.log(2) + (2 * n + 1) * np.log(alpha)
        return (- t3) ** n * np.exp(logs)

    terms2 = np.arange(1, N)
    def f2(n):
        logs = - np.log(factorial(n)) - n * np.log(2) + (2 * n - 1) * np.log(alpha)
        return 2* n * (- t3) ** n * np.exp(logs)

    return factor1 * np.sum(f1(terms1)) + factor2 * np.sum(f2(terms2))


def test_squeezed_fock_state():
    N = 150

    a = qt.destroy(N)
    psi0 = qt.basis(N, 0)

    rs = np.linspace(0.1, 1, 10)
    phis = np.linspace(0, 2 * np.pi, 10)

    for i, r in enumerate(rs):
        for j, phi in enumerate(phis):
            S = qt.squeeze(N, r * np.exp(1j * phi))
            psi = S @ a.dag() @ psi0
            #print(qt.expect(qt.qeye(N), analytical_squeezed_fock_state(r, phi, N)))
            print(psi.overlap(analytical_squeezed_fock_state(r, phi, N)))


def test_cat_state_squeezed_fock_state_overlap():
    N = 150

    a = qt.destroy(N)
    psi0 = qt.basis(N, 0)

    rs = np.linspace(0.1, 1, 10)
    phis = np.linspace(0, 2 * np.pi, 10)
    alphas = np.linspace(0.1, 1, 10)

    for i, r in enumerate(rs):
        for j, phi in enumerate(phis):
            S = qt.squeeze(N, r * np.exp(1j * phi))
            psi = S @ a.dag() @ psi0
            for k, alpha in enumerate(alphas):
                Dp = qt.displace(N, alpha)
                Dm = qt.displace(N, -alpha)
                cat = ((Dp - Dm) @ psi0).unit()
                num_overlap = cat.overlap(psi)
                print(num_overlap - analytical_cat_state_squeezed_fock_state_overlap(r, phi, alpha, N))


def test_doubly_squeezed_state():
    N = 150

    a = qt.destroy(N)
    psi0 = qt.basis(N, 0)

    rs = np.linspace(0.1, 1, 10)
    phis = np.linspace(0, 2 * np.pi, 10)

    for i, r1 in enumerate(rs):
        for j, phi1 in enumerate(phis):
            for k, r2 in enumerate(rs):
                for m, phi2 in enumerate(phis):
                    doubly_squeezed_state(r1, phi1, r2, phi2, N)


def test_doubly_squeezed_fock_state():
    N = 150

    psi0 = qt.basis(N, 0)
    a = qt.destroy(N)

    rs = np.linspace(0.1, 1, 10)
    phis = np.linspace(0, 2 * np.pi, 10)

    for i, r1 in enumerate(rs):
        for j, phi1 in enumerate(phis):
            S1 = qt.squeeze(N, - r1 * np.exp(1j * phi1))
            for k, r2 in enumerate(rs):
                for m, phi2 in enumerate(phis):
                    S2 = qt.squeeze(N, r2 * np.exp(1j * phi2))
                    psi1 = S1 @ S2 @ a.dag() @ psi0
                    print(psi1.overlap(doubly_squeezed_fock_state(r1, phi1, r2, phi2, N)))


def test_cat_state_doubly_squeezed_fock_state_overlap():
    N = 150

    psi0 = qt.basis(N, 0)
    a = qt.destroy(N)

    rs = np.linspace(0.1, 1, 10)
    phis = np.linspace(0, 2 * np.pi, 10)
    alphas = np.linspace(0.1, 1, 10)

    for i, r1 in enumerate(rs):
        for j, phi1 in enumerate(phis):
            print(f'i, j = {i, j}')
            S1 = qt.squeeze(N, - r1 * np.exp(1j * phi1))
            for k, r2 in enumerate(rs):
                for m, phi2 in enumerate(phis):
                    S2 = qt.squeeze(N, r2 * np.exp(1j * phi2))
                    for n, alpha in enumerate(alphas):
                        Dp = qt.displace(N, alpha)
                        Dm = qt.displace(N, -alpha)
                        cat = ((Dp - Dm) @ psi0).unit()
                        psi1 = S1 @ S2 @ a.dag() @ psi0
                        num_overlap = cat.overlap(psi1)
                        assert np.isclose(num_overlap - cat_state_doubly_squeezed_fock_state_overlap(r1, phi1, r2, phi2, alpha, N), 0)


if __name__ == '__main__':
    main()
