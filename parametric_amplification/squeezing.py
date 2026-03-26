"""
squeezing.py

A collection of utilities for simulating open second-order non-linear quantum
systems generating squeezed states. The script defines a ``SqueezingSystem``
class that can compute the output modes, the squeezed output quantum state
the squeezed output vacuum state, the input-output relation of the operators
and the E and F matrices used in this relation. Furthermore, three example
functions that plot Wigner functions for a coherent, cat, and Fock squeezed
state.

The code depends on QuTiP, TheWalrus, NumPy and SciPy packages. It is meant
to be run as a script (the ``main`` function calls the three demos) but the
individual functions and the ``SqueezingSystem`` class can also be imported
and reused in other projects.

For more information on the method behind this code package, see the original
work: {insert arXiv and journal link}
"""

import numpy as np
import qutip as qt
from thewalrus import quantum as twq
from scipy.special import factorial
import matplotlib as mpl
import matplotlib.pyplot as plt
from helper_functions import overlap
from opo import OpticalParametricOscillator

from typing import List, Callable, Tuple


def main():
    """Run the three example simulations sequentially."""
    w_coherent, w_sq_coherent = squeezed_coherent_state()
    w_cat, w_sq_cat = squeezed_cat_state()
    w_fock, w_sq_fock1, w_sq_fock2 = squeezed_fock_state()

    w_max = max(w_coherent.max(), w_cat.max(), w_sq_coherent.max(), w_sq_cat.max(),
                w_fock.max(), w_sq_fock1.max(), w_sq_fock2.max())
    w_min = min(w_coherent.min(), w_cat.min(), w_sq_coherent.min(), w_sq_cat.min(),
                w_fock.min(), w_sq_fock1.min(), w_sq_fock2.min())

    w_abs = max(abs(w_max), abs(w_min))

    # Plot the Wigner function
    nrm = mpl.colors.Normalize(w_min * w_abs / abs(w_min), w_max * w_abs / abs(w_max))

    fig, axs = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey=True)
    axs[0, 0].contourf(xvec, xvec, w_coherent, 100, cmap=mpl.cm.RdBu, norm=nrm)
    axs[0, 1].contourf(xvec, xvec, w_cat, 100, cmap=mpl.cm.RdBu, norm=nrm)
    axs[0, 2].contourf(xvec, xvec, w_fock, 100, cmap=mpl.cm.RdBu, norm=nrm)
    axs[1, 0].contourf(xvec, xvec, w_sq_coherent, 100, cmap=mpl.cm.RdBu, norm=nrm)
    axs[1, 1].contourf(xvec, xvec, w_sq_cat, 100, cmap=mpl.cm.RdBu, norm=nrm)
    axs[1, 2].contourf(xvec, xvec, w_sq_fock2, 100, cmap=mpl.cm.RdBu, norm=nrm)
    for i in range(2):
        for j in range(3):
            axs[i, j].set_aspect('equal')

    sm = mpl.cm.ScalarMappable(norm=nrm, cmap=mpl.cm.RdBu)
    sm.set_array([])  # required – the array itself isn’t used

    cbar = fig.colorbar(
        sm,
        ax=axs,  # attach to all contour axes
        location='right',
        shrink=1,  # make it a bit shorter than the full height
        pad=0.02,  # distance from the figure edge
    )
    plt.savefig('example_states_second_mode.png')
    plt.show()


def squeezed_coherent_state():
    """Generate and plot the Wigner function of a squeezed coherent state. In this example we do not use the full
    functionality of the method, but resort to calculating the Wigner function from the covariance matrix and
    displacement vector of the output squeezed coherent state, as in the example in the original work."""
    # Define the creation operator for the quantum state, here a displacement operator for the coherent state
    def f(au: qt.Qobj, audag: qt.Qobj):
        return (alpha * audag - np.conjugate(alpha) * au).expm()

    # Initialize the squeezing system with the system parameters
    ss = SqueezingSystem(f, M, u, fu, gu, zeta_u, xi_u, omegas)

    # Compute the output mode (only one when the input is a coherent state)
    v = ss.get_output_modes()[0]

    # Transform the output pulse by F and G
    zeta_v, xi_v, fv, gv = opo.get_fv_and_gv(omegas, v)

    # Copy of k, the length of the not normalized v, since it is needed for calculating the displacement vector
    # used in this special example, where the full functionality of the SqueezingSystem class is not used
    k = np.sqrt(alpha * np.conjugate(alpha) * (zeta_u ** 2 + xi_u ** 2)
                + 2 * zeta_u * xi_u * np.real(alpha ** 2 * overlap(gu, fu, omegas)))

    # Find the displacement vector
    r = np.array([np.sqrt(2)*(2 * zeta_u * np.real(alpha * overlap(v, fu, omegas)) - alpha * np.conjugate(alpha) / k),
                  0])

    # Calculate the covariance matrix of the coherent state. Since the covariance matrix is the same as the vacuum
    # state, we can use the function that generates the vacuum state covariance matrix
    cov = get_covariance_matrix(*ss.get_E_and_F(fv, gv, zeta_v, xi_v))

    # Compute the inverse covariance matrix and determinant of the covariance matrix to be used for computing the
    # Wigner function
    cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    # Shift the coordinates by the displacement vector and stack them for easy computation
    X, Y = np.meshgrid(xvec - r[0], xvec - r[1])
    diff = np.stack([X, Y], axis=-1)

    # Contract the inverse covariance matrix and the shifted coordinates, and then compute the Wigner function of
    # a Gaussian state.
    val = np.einsum('...i,ij,...j', diff, cov_inv, diff)
    w = np.real(np.exp(-val) / np.sqrt(det_cov) / np.pi)

    # Compute the wigner function of the initial coherent state
    cov0 = np.identity(2)
    r0 = np.array([np.sqrt(2) * np.real(alpha), np.sqrt(2) * np.imag(alpha)])
    X0, Y0 = np.meshgrid(xvec - r0[0], xvec - r0[1])
    diff0 = np.stack([X0, Y0], axis=-1)
    val0 = np.einsum('...i,ij,...j', diff0, cov0, diff0)
    w0 = np.real(np.exp(-val0) / np.pi)

    return w0, w


def squeezed_cat_state():
    """Generate and plot the Wigner function of a squeezed cat state. In this example we follow the full procedure
    of the original work, and show how the code package functionality can be used for each step. This generates the
    example of a squeezed cat state in the original work."""
    # Creation operator for a cat state
    def f(au: qt.Qobj, audag: qt.Qobj) -> qt.Qobj:
        """
        Creates a Schrödinger cat state in a given mode under the action upon the vacuum state

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
        D = (alpha * audag - np.conjugate(alpha) * au).expm()
        return (D + 1j * D.dag()) / np.sqrt(2)

    # Initialize the squeezing system with the system parameters
    ss = SqueezingSystem(f, M, u, fu, gu, zeta_u, xi_u, omegas)

    # Compute the output mode (only one when the input is this type of cat state)
    v = ss.get_output_modes()[0]

    # Transform the output pulse by F and G
    zeta_v, xi_v, fv, gv = opo.get_fv_and_gv(omegas, v)

    # Obtain the full density matrix of the squeezed cat state
    rho_squeezed_cat_state = ss.get_squeezed_output_state(v, fv, gv, zeta_v, xi_v)

    # Find the Wigner function numerically
    w = qt.wigner(rho_squeezed_cat_state, xvec, xvec)

    # Compute the wigner function of the initial cat state
    w0 = qt.wigner(f(a, a.dag()) @ vac, xvec, xvec)

    return w0, w


def squeezed_fock_state():
    """Generate and plot the Wigner function of a squeezed fock state. In this example we follow the full procedure
    of the original work, and show how the code package functionality can be used for each step. This generates the
    example of a squeezed fock state in the original work."""
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

    plt.figure()
    plt.plot(omegas, v1(omegas) * np.conjugate(v1(omegas)))
    plt.plot(omegas, v2(omegas) * np.conjugate(v2(omegas)))
    plt.xlim([-5.01, 5.01])
    plt.savefig('v1_v2.pdf')
    plt.show()

    # Transform the output pulses by F and G
    zeta_v1, xi_v1, zeta_v2, xi_v2, fv1, gv1, fv2, gv2 = opo.get_fv_and_gv(omegas, [v1, v2])

    # Obtain the full density matrix of the squeezed fock state
    rho_squeezed_fock_state = ss.get_squeezed_output_state([v1, v2], [fv1, fv2], [gv1, gv2],
                                                           [zeta_v1, zeta_v2], [xi_v1, xi_v2])

    # Find the Wigner function numerically of output mode 1
    w1 = qt.wigner(qt.ptrace(rho_squeezed_fock_state, 0), xvec, xvec)

    # Find the Wigner function numerically of output mode 2
    w2 = qt.wigner(qt.ptrace(rho_squeezed_fock_state, 1), xvec, xvec)

    # Compute the wigner function of the initial fock state
    w0 = qt.wigner(f(a, a.dag()) @ vac, xvec, xvec)

    return w0, w1, w2


class SqueezingSystem:
    """
    Implements the functionality for transforming a quantum state occupying a single input mode over a second-order
    non-linear quantum system, as described in {insert reference here} [1]

    Stores and computes the input and output of a second-order non-linear system that squeezes the input state into
    an output quantum state in one or two modes.

    The class stores the input mode and its transformation by F and G, and provides methods to obtain the output mode
    operators in terms of the input mode and in terms of E and F matrices, the squeezed vacuum state and the full
    squeezed output quantum state.

    Parameters
    ----------
    f : callable
        Function that maps a pair of annihilation/creation operators to a state creation operator that creates the
        input quantum state under the action upon the vacuum state.
    dim : int
        Dimension of the output Hilbert space(s). For a two mode output Hilbert space the same dimension will be used
        for both spaces. This is due to the limited functionality of The Walrus package used to compute the
        multidimensional Hermite polynomials.
    u : callable
        The input mode of the input quantum state in frequency.
    fu, gu : callable
        The resulting modes after integrating the input mode over the F and G functions; eq. 11 and 12 in the article.
    zeta_u, xi_u : float
        Normalisation constants for fu and gu modes.
    freq : ndarray
        Frequency grid over which the transformation is defined.
    """
    def __init__(
            self,
            f: Callable[[qt.Qobj, qt.Qobj], qt.Qobj],
            dim: int,
            u: Callable[[float | np.ndarray], complex | np.ndarray],
            fu: Callable[[float | np.ndarray], complex | np.ndarray],
            gu: Callable[[float | np.ndarray], complex | np.ndarray],
            zeta_u: float,
            xi_u: float,
            freq: np.ndarray,
    ):
        self.u = u
        self.freq = freq
        self.fu, self.gu, self.zeta_u, self.xi_u = fu, gu, zeta_u, xi_u
        self.dim = dim
        self.f = f

        # Precompute the values |alpha|^2 = <a^dag a> and beta = <a * a>
        self.alpha_sq, self.beta = self.__get_expectation_values()

        # Determine whether the system satisfies the single‑mode condition |alpha|^2 = |beta| (Eq. 9 in ref. [1]).
        # This influences which output‑mode formulas are used later.
        self.is_single_mode = self.__check_output_single_mode()

    def __get_expectation_values(self):
        """
        Compute <a^dag a> and <a*a> for the input state.
        """
        vac = qt.basis(self.dim, 0)
        a = qt.destroy(self.dim)
        psi = self.f(a, a.dag()) @ vac
        adaga = qt.expect(a.dag() @ a, psi)
        aa = qt.expect(a @ a, psi)
        return adaga, aa

    def get_output_modes(self):
        """
        Return the output modes of the system.

        For an input state obeying the single mode condition (eq. 9 of ref. [1]) a single callable [v] is returned.
        For a general input state not obeying this condition (eq. 9 of ref. [1]) two callables [v1, v2] are returned.
        """
        # Overlap between gu and fu used in the calculation
        gu_fu = overlap(self.gu, self.fu, self.freq)

        # Single mode output mode (eq. 10 in [1])
        if self.is_single_mode:
            alpha_ = np.sqrt(self.beta)
            k = np.sqrt(alpha_ * np.conjugate(alpha_) * (self.zeta_u ** 2 + self.xi_u ** 2)
                        + 2 * self.zeta_u * self.xi_u * np.real(self.beta * gu_fu))
            v = lambda omega: (alpha_ * self.zeta_u * self.fu(omega)
                               + np.conjugate(alpha_) * self.xi_u * self.gu(omega)) / k
            return [v]

        # Two output modes computed as in app. A of [1]
        else:
            # Rename gu_fu to keep the notation of the paper
            varepsilon = np.conjugate(gu_fu)
            delta = np.sqrt(1 - varepsilon * np.conjugate(varepsilon))

            # Construct the orthogonal part of gu with respect to fu
            hu = lambda omega: (self.gu(omega) - varepsilon * self.fu(omega)) / delta

            # Define the variables used in the appendix
            x = (self.alpha_sq * self.zeta_u ** 2
                 + 2 * self.zeta_u * self.xi_u * np.real(np.conjugate(self.beta) * varepsilon)
                 + self.alpha_sq * self.xi_u ** 2 * varepsilon * np.conjugate(varepsilon))
            y = self.beta * self.zeta_u * self.xi_u * delta + self.alpha_sq * self.xi_u ** 2 * varepsilon * delta
            z = self.alpha_sq * self.xi_u ** 2 * delta ** 2

            # Compute the coefficients of the fu and gu modes in v1 and calculate v1
            c1_fu = (x - z + np.sqrt((x - z) ** 2 + 4 * y * np.conjugate(y)))
            c1_hu = 2 * np.conjugate(y)
            v1_norm = np.sqrt(2 * ((x - z) ** 2 + 4 * y * np.conjugate(y)
                                   + (x - z) * np.sqrt((x - z) ** 2 + 4 * y * np.conjugate(y))))
            v1 = lambda omega: (c1_fu * self.fu(omega) + c1_hu * hu(omega)) / v1_norm

            # Compute the coefficients of the fu and gu modes in v2 and calculate v2
            c2_fu = (x - z - np.sqrt((x - z) ** 2 + 4 * y * np.conjugate(y)))
            c2_hu = 2 * np.conjugate(y)
            v2_norm = np.sqrt(2 * ((x - z) ** 2 + 4 * y * np.conjugate(y)
                                   - (x - z) * np.sqrt((x - z) ** 2 + 4 * y * np.conjugate(y))))
            v2 = lambda omega: (c2_fu * self.fu(omega) + c2_hu * hu(omega)) / v2_norm
            return [v1, v2]

    def get_squeezed_output_state(
            self,
            v: Callable[[float], complex] | List[Callable[[float], complex]],
            fv: Callable[[float], complex] | List[Callable[[float], complex]],
            gv: Callable[[float], complex] | List[Callable[[float], complex]],
            zeta_v: float | List[float],
            xi_v: float | List[float]
    ) -> qt.Qobj:
        """
        Compute the density matrix of the squeezed output state using the method described in ref. [1] in eq. 32.
        Finds the input operator au in terms of output operators bv1, bv2, and the squeezed vacuum state in these
        modes. Then uses the user defined f-operator to create the output squeezed state in the output modes.

        Parameters
        ----------
        v : callable or list[callable]
            Output mode operator(s) returned by ``get_output_modes``.
        fv, gv : callable or list[callable]
            Modes after transforming the output modes v by F and G.
        zeta_v, xi_v : float or list[float]
            Normalisation constants for the fv and gv modes.

        Returns
        -------
        qt.Qobj
            The full (possibly multimode) density matrix after the second-order non-linear transformation.
        """
        au: qt.Qobj = self.decompose_input_operator(v)
        squeezed_vacuum: qt.Qobj = self.get_squeezed_vacuum_state(fv, gv, zeta_v, xi_v)
        return self.f(au, au.dag()) @ squeezed_vacuum @ self.f(au, au.dag()).dag()

    def decompose_input_operator(
            self,
            v: Callable[[float], complex] | List[Callable[[float], complex]]
    ) -> qt.Qobj:
        """
        Computes the input-output operator relation of eq. 6 (eq. 16/17 for a single output mode, eq. 21 otherwise) of
        ref. [1].

        The result au is a QuTiP operator that can be fed into self.f.

        Parameters
        ----------
        v : callable or list[callable]
            Output mode operator(s) returned by ``get_output_modes``.

        Returns
        -------
        qt.Qobj
            The input operator expressed in terms of output operators.
        """
        # Single mode case
        if self.is_single_mode:
            fu_v = overlap(self.fu, v, self.freq)
            v_gu = overlap(v, self.gu, self.freq)
            bv: qt.Qobj = qt.destroy(self.dim)
            # Eq. 16 of [1] without the h mode as it is irrelevant for the transformation
            au = self.zeta_u * fu_v * bv - self.xi_u * v_gu * bv.dag()

        # Two mode case
        else:
            v1, v2 = v
            fu_v1 = overlap(self.fu, v1, self.freq)
            v1_gu = overlap(v1, self.gu, self.freq)
            fu_v2 = overlap(self.fu, v2, self.freq)
            v2_gu = overlap(v2, self.gu, self.freq)
            bv1 = qt.tensor(qt.destroy(self.dim), qt.qeye(self.dim))
            bv2 = qt.tensor(qt.qeye(self.dim), qt.destroy(self.dim))
            # Eq. 21 of [1]
            au = (self.zeta_u * fu_v1 * bv1 - self.xi_u * v1_gu * bv1.dag()
                  + self.zeta_u * fu_v2 * bv2 - self.xi_u * v2_gu * bv2.dag())
        return au

    def __check_output_single_mode(self) -> bool:
        """
        Evaluate whether the input state satisfies the single‑mode criterion <a^dag a> = |<a*a>| (Eq. 9 of [1])
        Returns True if the condition holds and the output is single‑mode, False otherwise.
        """
        return np.isclose(self.alpha_sq ** 2, self.beta * np.conjugate(self.beta))

    def get_squeezed_vacuum_state(
            self,
            fv: Callable[[float], complex] | List[Callable[[float], complex]],
            gv: Callable[[float], complex] | List[Callable[[float], complex]],
            zeta_v: float,
            xi_v: float
    ) -> qt.Qobj:
        """
        Construct the squeezed vacuum state in the v mode(s) using the multidimensional Hermite polynomials.

        Parameters
        ----------
        fv, gv : callable or list[callable]
            Modes after transforming the output modes v by F and G.
        zeta_v, xi_v : float or list[float]
            Normalisation constants for the fv and gv modes.

        Returns
        -------
        qt.Qobj
            A density matrix representation of the squeezed vacuum state in the output mode(s).
        """
        # Obtain the E and F matrices for the mode transformation
        E, F = self.get_E_and_F(fv, gv, zeta_v, xi_v)

        # Compute the covariance matrix of the squeezed vacuum state based on the operator transformation
        cov = get_covariance_matrix(E, F)

        # Single mode case builds a 2x2 covariance matrix and uses it to compute the multidimensional Hermite
        # polynomials implemented in The Walrus to find the squeezed vacuum state in fock space as in eq. 24 of ref. [1]
        if self.is_single_mode:
            density_matrix = qt.Qobj(twq.density_matrix(mu=np.array([0, 0]), cov=cov / 2,
                                                        cutoff=self.dim, normalize=True, hbar=1))

        # Two mode case builds a 4x4 covariance matrix and uses it to compute the multidimensional Hermite polynomials
        # implemented in The Walrus to find the squeezed vacuum state in fock space as in eq. 24 of ref. [1]
        else:
            density_matrix_array = twq.density_matrix(mu=np.array([0, 0, 0, 0]), cov=cov / 2,
                                                      cutoff=self.dim, hbar=1)

            # Convert the density matrix from The Walrus Hilbert space dimension formalism to QuTiP's formalism
            density_matrix = convert_density_matrix(density_matrix_array)
        return density_matrix

    def get_E_and_F(
            self,
            fv: Callable[[float], complex] | List[Callable[[float], complex]],
            gv: Callable[[float], complex] | List[Callable[[float], complex]],
            zeta_v: float,
            xi_v: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the operator transformation E and F matrices that capture the mode transformation as in eq. 22 of [1].
        These matrices are used for calculating the covariance matrix of the squeezed vacuum state. The computation of
        these matrices are described in appendix B of [1]

        Parameters
        ----------
        fv, gv : callable or list[callable]
            Modes after transforming the output modes v by F and G.
        zeta_v, xi_v : float or list[float]
            Normalisation constants for the fv and gv modes.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two arrays for the E and the F matrix in that order.
        """
        # The single mode case
        if self.is_single_mode:
            fv_u = overlap(fv, self.u, self.freq)
            u_gv = overlap(self.u, gv, self.freq)
            # E and F given in eq. 70 and 71 of [1]
            E = np.array([[zeta_v * fv_u, zeta_v * np.sqrt(1 - fv_u * np.conjugate(fv_u))]])
            F = np.array([[xi_v * u_gv, zeta_v * np.sqrt(1 - fv_u * np.conjugate(fv_u))]])

        # The two mode case
        else:
            # Unpack the modes
            fv1, fv2 = fv
            gv1, gv2 = gv
            zeta_v1, zeta_v2 = zeta_v
            xi_v1, xi_v2 = xi_v

            # Find the ancillary modes
            fv1_u = overlap(fv1, self.u, self.freq)
            t = lambda omega: ((fv1(omega) - np.conjugate(fv1_u) * self.u(omega))
                               / np.sqrt(1 - fv1_u * np.conjugate(fv1_u)))
            fv2_u = overlap(fv2, self.u, self.freq)
            fv2_t = overlap(fv2, t, self.freq)
            s = lambda omega: ((fv2(omega) - np.conjugate(fv2_u) * self.u(omega) - np.conjugate(fv2_t) * t(omega))
                               / np.sqrt(1 - fv2_u * np.conjugate(fv2_u) - fv2_t * np.conjugate(fv2_t)))

            # Compute the overlap between fv1 and fv2 with the input mode and ancillary modes
            fv1_modes = np.array([fv1_u, overlap(fv1, t, self.freq), overlap(fv1, s, self.freq)])
            fv2_modes = np.array([fv2_u, fv2_t, overlap(fv2, s, self.freq)])

            # Compute the overlap between gv1 and gv2 with the input mode and ancillary modes
            modes_gv1 = np.array([overlap(self.u, gv1, self.freq), overlap(t, gv1, self.freq),
                                  overlap(s, gv1, self.freq)])
            modes_gv2 = np.array([overlap(self.u, gv2, self.freq), overlap(t, gv2, self.freq),
                                  overlap(s, gv2, self.freq)])

            # E and F given in eq. 64 and 65 of [1]
            E = np.block([[zeta_v1 * fv1_modes], [zeta_v2 * fv2_modes]])
            F = np.block([[xi_v1 * modes_gv1], [xi_v2 * modes_gv2]])
        return E, F


def get_covariance_matrix(E: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Build the covariance matrix for the (single‑ or two‑mode) squeezed vacuum state.

    The matrix is constructed from the E and F matrices defining the input-output transformation on the levels of
    operators defined in eq. 22 of [1].

    Parameters
    ----------
    E, F : np.ndarray
        Matrices E and F of the input-output transformation of operators.

    Returns
    -------
    np.ndarray
        A ``(2N, 2N)`` real covariance matrix of the output squeezed vacuum state.
    """
    dims = E.shape
    dims_F = F.shape
    if not dims == dims_F:
        raise Exception('Dimension of E and F must be the same')

    N = dims[0]
    Identity = np.eye(2 * N)

    EFT = E @ F.T
    FFdag = F @ F.conjugate().T

    cov_2 = np.block([[np.real(EFT + FFdag), np.imag(EFT - FFdag)],
                      [np.imag(EFT + FFdag), np.real(FFdag - EFT)]])

    return Identity + 2 * cov_2


def convert_density_matrix(density_matrix: np.ndarray):
    """
    Converts a density matrix that is stored as a 4‑D array (density_matrix[n1, m1, n2, m2]) into a 2‑D matrix
    density_matrix[i, j] where

        i = n1 * N2 + n2
        j = m1 * N2 + m2

    This conversion is necessary as the convention for Hilbert space dimensions is not the same for The Walrus library
    and QuTiP library. This function converts from The Walrus convention to QuTiP convention.

    Parameters
    ----------
    density_matrix : np.ndarray
        Shape (N1, N1, N2, N2) in The Walrus convention.

    Returns
    -------
    qt.Qobj
        A Qobj whose underlying data is the reshaped density matrix in QuTiPs convention.
    """

    N = density_matrix.shape[0]
    dm_transposed = density_matrix.transpose(0, 2, 1, 3)
    new_density_matrix = dm_transposed.reshape(N * N, N * N)
    return qt.Qobj(new_density_matrix, dims=[[N, N], [N, N]])


if __name__ == '__main__':
    # Physical parameters
    gamma = 1  # decay rate of the open quantum system
    xi = 0.1j  # amplitude strength of the parametric oscillator
    alpha = 1 + 1j  # displacement of the coherent state
    n = 1  # Number of photons in the fock state
    # tp    = 4           # pulse center in time of input gaussian pulse
    # tau   = 1           # pulse width in time of input gaussian pulse
    Gamma = 1
    Delta = 0  # detuning between the parametric oscillators and the input pulse
    M = 100  # the Hilbert space size of the output numerical quantum state

    # Array of frequencies for the relevant spectrum
    # omegas = np.linspace(-4, 4, 1000)
    # Array of frequencies for the relevant spectrum
    omegas = np.linspace(-300, 300, 30000)

    # A gaussian input pulse in frequency domain (fourier transform of time domain)
    # u = lambda omega: np.sqrt(tau) / np.pi ** (1 / 4) * np.exp(-tau ** 2 / 2 * omega ** 2 + 1j * tp * omega)
    # A one-sided exponential input pulse in frequency domain
    u = lambda omega: np.sqrt(Gamma / (2 * np.pi)) * (1j * omega + Gamma / 2) / (omega ** 2 + Gamma ** 2 / 4)

    plt.figure()
    plt.plot(omegas, u(omegas) * np.conjugate(u(omegas)))
    plt.xlim([-5.01, 5.01])
    plt.savefig('u.pdf')
    plt.show()

    # Create an OPO with the given parameters
    opo = OpticalParametricOscillator(gamma, xi, Delta)

    # Transform the input pulse by F and G
    zeta_u, xi_u, fu, gu = opo.get_fu_and_gu(omegas, u)

    # Define a grid for plotting the Wigner function
    xvec = np.linspace(-5, 5, 200)

    # Define the input operator and vacuum state
    a = qt.destroy(M)
    vac = qt.basis(M, 0)

    main()
