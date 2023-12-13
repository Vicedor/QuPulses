import numpy as np
import qutip as qt
from scipy.integrate import quad
import math
from cmath import phase
from math import factorial
from scipy.linalg import sqrtm
from typing import Callable, Tuple, List


def overlap(f, g, xs):
    """
    Calculates the inner product between functions f and g given by <f|g>
    :param f: Leftmost function in the inner product
    :param g: Rightmost function in the inner product
    :param xs: The axis which the functions are defined over
    :return: The value of the inner product
    """
    return quad(lambda omega: np.conjugate(f(omega)) * g(omega), xs[0], xs[-1], complex_func=True)[0]


def create_bs_interaction(a: qt.Qobj, b: qt.Qobj, theta: float, phi: float) -> qt.Qobj:
    """
    Creates a beam-splitter unitary using the qutip expm-method
    :param a: The annihilation operator for the first input mode to the beam-splitter
    :param b: The annihilation operator for the second input mode to the beam-splitter
    :param theta: The rotation angle for the beam-splitter
    :param phi: The phase difference for the beam-splitter
    :return: A qutip unitary of the beam-splitter operator
    """
    return (1j * theta * (a.dag() * b * np.exp(1j * phi) + a * b.dag() * np.exp(-1j * phi))).expm(method='sparse')


def rref(matrix: np.ndarray) -> np.ndarray:
    """
    Performs the decomposition of a matrix to reduced row echelon form
    :param matrix: The matrix to decompose
    :return: The decomposed input matrix to reduced row echelon form
    """
    for i in range(matrix.shape[0]):
        for j in range(i + 1):
            if i == j:
                matrix[i, :] = matrix[i, :] / matrix[i, j]
            else:
                matrix[i, :] = matrix[i, :] - matrix[i, j] / matrix[j, j] * matrix[j, :]
    return matrix


def decompose_unitary(U: np.ndarray) -> List[np.ndarray]:
    """
    decomposes a unitary matrix to a number of 2x2 unitaries. This decomposition follows closely the procedure in
    Nielsen & Chuang.
    :param U: The unitary to be decomposed
    :return: An array of unitary 2x2 matrices, whose product gives U.
    """
    d = U.shape[0]
    Us = []
    for i in range(d - 2):
        for j in range(i + 1, d):
            Ui = np.identity(d, dtype=np.complex_)
            a = U[i, i]
            b = U[j, i]
            if np.isclose(b, 0 + 0j):
                if i == d - 1:
                    Ui[i, i] = a.conjugate()
            else:
                c = np.sqrt(a * a.conjugate() + b * b.conjugate())
                Ui[i, i] = a.conjugate() / c
                Ui[i, j] = b.conjugate() / c
                Ui[j, i] = - b / c
                Ui[j, j] = a / c
            Us.append(Ui.conjugate().T)
            U = Ui @ U
    Ui = np.identity(d, dtype=np.complex_)
    Ui[d - 2, d - 2] = U[d - 2, d - 2].conjugate()
    Ui[d - 2, d - 1] = U[d - 1, d - 2].conjugate()
    Ui[d - 1, d - 2] = U[d - 2, d - 1].conjugate()
    Ui[d - 1, d - 1] = U[d - 1, d - 1].conjugate()
    Us.append(Ui.conjugate().T)
    return Us


def decompose_W_e(W_e: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Decomposes W_E^dagger into 2x2 rotations using the decompose_unitary function, and then finds the rotation
    parameters for the beam-splitter relations. It is not necessary to decompose U, as only the first mode is
    non-vacuum and thus it is only necessary to mix that mode with the other vacuum modes. The parameters for these
    transformations can be read directly from U without any decomposition.
    :param W_e: The W_e^dagger matrix to be decomposed.
    :return:
    """
    n = W_e.shape[0]
    Us = decompose_unitary(W_e)
    m = factorial(n - 1)
    assert m == len(Us)
    thetas: List[float] = [0 for _ in range(m)]
    phis: List[float] = [0 for _ in range(m)]

    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            thetas[k] = np.arccos(np.abs(Us[k][i, i]))
            phis[k] = -phase(1j * Us[k][j, i]) - phase(Us[k][i, i])
            if math.isnan(thetas[k]):
                thetas[k] = 0
            k += 1

    return thetas, phis


def bloch_messiah(E: np.ndarray, F: np.ndarray, atol=1e-3, rtol=1e-3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs the Bloch-Messiah decomposition as in the paper "Reexamination of Bloch-Messiah reduction" (2016) by
    Gianfranco Cariolaro and Gianfranco Pierobon. The naming of variables follow closely their notation
    :param E: The E matrix from the paper
    :param F: The F matrix from the paper
    :param atol: The absolute tolerance for the assertions that the decompositions are performed correctly
    :param rtol: The relative tolerance for the assertions that the decompositions are performed correctly
    :return: The scriptU, Lambda_E and scriptW_E matrices from the paper (eq. 34)
    """
    # Step 1 and 2
    lambda_E_sq, U = np.linalg.eig(E @ E.conjugate().T)

    lambda_E = np.diag(np.sqrt(lambda_E_sq))
    lambda_F = np.sqrt(U.conjugate().T @ F @ F.conjugate().T @ U)

    Vh_E = np.linalg.inv(lambda_E) @ U.conjugate().T @ E
    W_E = Vh_E.conjugate().T
    W_F = (np.linalg.inv(lambda_F) @ U.conjugate().T @ F).conjugate().T

    assert np.isclose(U @ lambda_E @ W_E.conjugate().T, E, atol=atol, rtol=rtol).all()
    assert np.isclose(U @ lambda_F @ W_F.conjugate().T, F, atol=atol, rtol=rtol).all()

    # Step 3

    G = W_E.conjugate().T @ W_F.conjugate()

    D = sqrtm(G)

    # Step 5

    scriptW_E = W_E @ D
    scriptU = U @ D
    scriptW_F = W_F @ D

    assert np.all(np.isclose(scriptU @ lambda_E @ scriptW_E.conjugate().T, E, atol=atol, rtol=rtol))
    assert np.all(np.isclose(scriptU @ lambda_F @ scriptW_F.conjugate().T, F, atol=atol, rtol=rtol))
    assert np.all(np.isclose(scriptW_F, scriptW_E.conjugate(), atol=atol, rtol=rtol))
    return scriptU, lambda_E, scriptW_E


class SingleModeBlochMessiah:
    """
    A class that can find the output state of a single mode in the output field of an open quantum system governed
    by a mode transformation as in eq. 3 of the main paper.
    """
    def __init__(self, u: Callable[[float], float], f: Callable[[float], float], g: Callable[[float], float],
                 xs: np.ndarray, N: int, psi0: qt.Qobj, atol=1e-3, rtol=1e-3):
        """
        Initializes the class
        :param u: The input function u(omega) or u(t) as a Callable function
        :param f: The f function from the main paper given as the integral over F and v (see just after eq. 7)
        :param g: The g function from the main paper given as the integral over G and v (see just after eq. 7)
        :param xs: The axis over which the functions u, f and g are defined
        :param N: The size of the Hilbert space for the output mode calculation
        :param psi0: The initial state of the input in the u-mode
        :param atol: The absolute tolerance for the assertions along the way
        :param rtol: The relative tolerance for the assertions along the way
        """
        self._u: Callable[[float], float] = u
        self._f_temp: Callable[[float], float] = f
        self._g_temp: Callable[[float], float] = g
        self._xs: np.ndarray = xs
        self._N = N
        self._psi0 = psi0
        self._atol = atol
        self._rtol = rtol

    def get_output_state(self) -> qt.Qobj:
        """
        Computes the output state of the requested output mode
        :return: A qutip state of the output mode
        """
        coefs, theta_vac = self._get_mode_coefs()
        E_mat, F_mat = self._get_transformation_matrix(coefs)
        scriptU, lambda_E, scriptW_E = bloch_messiah(E_mat, F_mat, atol=self._atol, rtol=self._rtol)
        rs, thetas, phis = self._get_parameters(scriptU, lambda_E, scriptW_E)
        rhov = self._transform_state(*rs, *thetas, *phis, theta_vac)
        return rhov

    def _get_mode_coefs(self) -> Tuple[List[float], float]:
        """
        Computes the coefficients for the modes as in eq. 8 of the main paper
        :return: A list of the coefficients, and the angle for mixing with the final vacuum mode.
        """
        zeta = np.sqrt(overlap(self._f_temp, self._f_temp, self._xs))
        xi = np.sqrt(overlap(self._g_temp, self._g_temp, self._xs))

        f = lambda x: self._f_temp(x) / zeta
        g = lambda x: self._g_temp(x) / xi

        uf = overlap(self._u, f, self._xs)
        ug = overlap(self._u, g, self._xs)

        h = lambda omega: (f(omega) - self._u(omega) * uf) / np.sqrt(1 - uf * uf.conjugate())
        k = lambda omega: (g(omega) - self._u(omega) * ug) / np.sqrt(1 - ug * ug.conjugate())

        kh = overlap(k, h, self._xs)

        A1 = zeta * uf.conjugate()
        B1 = xi * ug
        C1 = zeta * np.sqrt(1 - uf.conjugate()*uf) * kh.conjugate()
        D1 = xi * np.sqrt(1 - ug.conjugate() * ug)

        #print('target:', A1.conjugate() * A1 + 2 * B1.conjugate() * B1 + D1.conjugate() * D1)
        norm = np.sqrt(A1.conjugate() * A1 - B1.conjugate() * B1 + C1.conjugate() * C1 - D1.conjugate() * D1)
        theta_vac = np.arccos(norm)
        A1 = A1 / norm
        B1 = B1 / norm
        C1 = C1 / norm
        D1 = D1 / norm
        return [A1, C1, B1, D1], theta_vac

    def _get_transformation_matrix(self, coefs: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the E and F matrices by introducing an ancillary mode. The coefficients of the ancillary mode are not
        unique, so one solution out of many is found.
        :param coefs: The coefficients for the actual output mode given in eq. 8 in the main paper
        :return: Matrices E and F from G. Cariolaro and G. Pierobon (2016)
        """
        A1, C1, B1, D1 = coefs

        t1 = (D1 * B1.conjugate() - A1 * C1.conjugate())
        t2 = (A1 * D1.conjugate() - C1 * B1.conjugate())
        t3 = (A1 * A1.conjugate() - B1 * B1.conjugate())
        t4 = (D1 * A1.conjugate() - B1 * C1.conjugate())
        t5 = (B1 * D1.conjugate() - C1 * A1.conjugate())

        C2 = 1
        D2 = 0
        B2 = (C2 * t4 + D2 * t5) / t3
        A2 = (C2 * t1 + D2 * t2) / t3

        norm = np.sqrt(A2.conjugate() * A2 - B2.conjugate() * B2 + C2.conjugate() * C2 - D2.conjugate() * D2)

        if np.imag(norm) != 0:
            C2 = 0
            D2 = 1
            B2 = (C2 * t4 + D2 * t5) / t3
            A2 = (C2 * t1 + D2 * t2) / t3
            norm = np.sqrt(A2.conjugate() * A2 - B2.conjugate() * B2 + C2.conjugate() * C2 - D2.conjugate() * D2)

        A2 = A2 / norm
        B2 = B2 / norm
        C2 = C2 / norm
        D2 = D2 / norm

        coefs2 = [A2, C2, B2, D2]

        E = np.array([coefs[:2], coefs2[:2]], dtype=np.complex_)
        F = np.array([coefs[2:], coefs2[2:]], dtype=np.complex_)

        assert np.isclose(E @ E.conjugate().T - F @ F.conjugate().T, np.identity(2), atol=self._atol, rtol=self._rtol).all()
        assert np.isclose(E @ F.T, F @ E.T, atol=self._atol, rtol=self._rtol).all()
        return E, F

    def _get_parameters(self, scriptU: np.ndarray, lambda_E: np.ndarray,
                       scriptW_E: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Finds the parameters for the rotation and squeezing transformations, as described in the main paper eq. 9
        :param scriptU: The first rotation matrix from G. Cariolaro and G. Pierobon (2016) eq. 34.
        :param lambda_E: The squeezing matrix from G. Cariolaro and G. Pierobon (2016) eq. 34.
        :param scriptW_E: The second rotation matrix from G. Cariolaro and G. Pierobon (2016) eq. 34.
        :return: The parameters for the rotation operators and squeezing operators in the order
        (squeezing 1, squeezing 2), (rotation 1, rotation 2), (phase 1, phase 2)
        """
        # Squeezing parameters
        r1 = np.arccosh(lambda_E[0, 0])
        r2 = np.arccosh(lambda_E[1, 1])

        # Rotation parameters
        theta1 = np.arccos(np.abs(scriptW_E.conjugate().T[0, 0]))
        phi1 = -phase(1j * scriptW_E.conjugate().T[1, 0]) - phase(scriptW_E.conjugate().T[0, 0])
        theta2 = np.arccos(np.abs(scriptU[0, 0]))
        phi2 = phase(1j * scriptU[0, 1]) - phase(scriptU[0, 0])
        return (r1, r2), (theta1, theta2), (phi1, phi2)

    def _transform_state(self, r1, r2, theta1, theta2, phi1, phi2, theta_vac) -> qt.Qobj:
        """
        Transforms the input quantum state to the output quantum state using the parameters from the Bloch-Messiah
        decomposition.
        :param r1: Squeezing parameter for mode 1
        :param r2: Squeezing parameter for mode 2
        :param theta1: Rotation parameter for first rotation
        :param theta2: Rotation parameter for second rotation
        :param phi1: Phase parameter for first rotation
        :param phi2: Phase parameter for second rotation
        :param theta_vac: Rotation angle for mixing with vacuum mode in the end
        :return: The output quantum state as a qutip state
        """
        a = qt.destroy(self._N)
        I = qt.qeye(self._N)

        # Define quantum system
        au = qt.tensor(a, I)
        a1 = qt.tensor(I, a)

        rho_u = qt.ket2dm(self._psi0)
        rho_1 = qt.ket2dm(qt.basis(self._N, 0))
        rho_2 = qt.ket2dm(qt.basis(self._N, 0))

        rho = qt.tensor(rho_u, rho_1)

        # Rotate
        U_au_a1 = create_bs_interaction(au, a1, theta1, phi1)
        rho_t = U_au_a1 * rho * U_au_a1.dag()

        # Squeeze
        S1: qt.Qobj = qt.tensor(qt.squeeze(self._N, -r1), I)
        S2: qt.Qobj = qt.tensor(I, qt.squeeze(self._N, -r2))
        rho_t = S2 * S1 * rho_t * S1.dag() * S2.dag()

        # Rotate
        V_au_a1 = create_bs_interaction(au, a1, theta2, phi2)
        rho_t = V_au_a1 * rho_t * V_au_a1.dag()

        # Mix with vacuum
        rhov = rho_t.ptrace(0)
        rhov2 = qt.tensor(rhov, rho_2)
        V_a1_a4 = create_bs_interaction(au, a1, theta_vac, 0)

        # Trace over desired output mode
        rhov = (V_a1_a4 * rhov2 * V_a1_a4.dag()).ptrace(0)
        print(qt.expect(a.dag()*a, rhov))
        return rhov


class TwoModeBlochMessiah:
    """
    A class that can find the combined output state of two modes in the output field of an open quantum system governed
    by a mode transformation as in eq. 3 of the main paper.
    """
    def __init__(self, u: Callable[[float], float],
                 f1: Callable[[float], float], f2: Callable[[float], float],
                 g1: Callable[[float], float], g2: Callable[[float], float],
                 xs: np.ndarray, N: int, psi0: qt.Qobj, atol=1e-3, rtol=1e-3):
        """
        Initializes the class. v1 and v2 are the two orthogonal output modes for which the output quantum state is
        desired.
        :param u: The input function u(omega) or u(t) as a Callable function
        :param f1: The f function from the main paper given as the integral over F and v1 (see just after eq. 7)
        :param f2: The f function from the main paper given as the integral over F and v2 (see just after eq. 7)
        :param g1: The g function from the main paper given as the integral over G and v1 (see just after eq. 7)
        :param g2: The g function from the main paper given as the integral over G and v2 (see just after eq. 7)
        :param xs: The axis over which the functions u, f1, f2, g1 and g are defined
        :param N: The size of the Hilbert space for the output mode calculation
        :param psi0: The initial state of the input in the u-mode
        :param atol: The absolute tolerance for the assertions along the way
        :param rtol: The relative tolerance for the assertions along the way
        """
        self._u: Callable[[float], float] = u
        self._f1_temp: Callable[[float], float] = f1
        self._f2_temp: Callable[[float], float] = f2
        self._g1_temp: Callable[[float], float] = g1
        self._g2_temp: Callable[[float], float] = g2
        self._xs: np.ndarray = xs
        self._N: int = N
        self._psi0: qt.Qobj = psi0
        self._atol = atol
        self._rtol = rtol

    def get_output_state(self) -> qt.Qobj:
        """
        Computes the output state of the requested output mode
        :return: A qutip state of the output mode
        """
        coefs1, coefs2, theta_vac = self._get_mode_coefs()
        E_mat, F_mat = self._get_transformation_matrix(coefs1, coefs2)
        scriptU, lambda_E, scriptW_E = bloch_messiah(E_mat, F_mat, atol=self._atol, rtol=self._rtol)
        rs, thetas, phis = self._get_parameters(scriptU, lambda_E, scriptW_E)
        rhov1v2 = self._transform_state(*rs, *thetas, *phis, theta_vac)
        return rhov1v2

    def _get_mode_coefs(self) -> Tuple[List[float], List[float], float]:
        """
        Computes the coefficients for the modes as in eq. 8 of the main paper for mode 1, and for mode 2 eq. 8 is
        also calculated, but then a further decomposition is performed to make it orthogonal to mode 1 as well.
        :return: Two lists of the coefficients of mode 1 and mode 2, and the angle for mixing with the final vacuum
        mode.
        """
        zeta1 = np.sqrt(overlap(self._f1_temp, self._f1_temp, self._xs))
        xi1 = np.sqrt(overlap(self._g1_temp, self._g1_temp, self._xs))

        zeta2 = np.sqrt(overlap(self._f2_temp, self._f2_temp, self._xs))
        xi2 = np.sqrt(overlap(self._g2_temp, self._g2_temp, self._xs))

        f1 = lambda omega: self._f1_temp(omega) / zeta1
        g1 = lambda omega: self._g1_temp(omega) / xi1

        f2 = lambda omega: self._f2_temp(omega) / zeta2
        g2 = lambda omega: self._g2_temp(omega) / xi2

        """ Getting all functions for v1 mode decomposition """

        uf1 = overlap(self._u, f1, self._xs)
        ug1 = overlap(self._u, g1, self._xs)

        h1 = lambda omega: (f1(omega) - self._u(omega) * uf1) / np.sqrt(1 - uf1 * uf1.conjugate())
        k1 = lambda omega: (g1(omega) - self._u(omega) * ug1) / np.sqrt(1 - ug1 * ug1.conjugate())

        k1h1 = overlap(k1, h1, self._xs)

        s1 = lambda omega: (h1(omega) - k1(omega) * k1h1) / np.sqrt(1 - k1h1 * k1h1.conjugate())

        """ Getting all functions for v2 mode decomposition """

        uf2 = overlap(self._u, f2, self._xs)
        ug2 = overlap(self._u, g2, self._xs)

        h2 = lambda omega: (f2(omega) - self._u(omega) * uf2) / np.sqrt(1 - uf2 * uf2.conjugate())
        k2 = lambda omega: (g2(omega) - self._u(omega) * ug2) / np.sqrt(1 - ug2 * ug2.conjugate())

        k2h2 = overlap(k2, h2, self._xs)
        s2 = lambda omega: (h2(omega) - k2(omega) * k2h2) / np.sqrt(1 - k2h2 * k2h2.conjugate())

        # Diagonalize with respect to 1-modes
        k1k2 = overlap(k1, k2, self._xs)
        k3 = lambda omega: (k2(omega) - k1(omega) * k1k2) / np.sqrt(1 - k1k2 * k1k2.conjugate())

        s1k3 = overlap(s1, k3, self._xs)
        k4 = lambda omega: (k3(omega) - s1(omega) * s1k3) / np.sqrt(1 - s1k3 * s1k3.conjugate())

        k1s2 = overlap(k1, s2, self._xs)
        s3 = lambda omega: (s2(omega) - k1(omega) * k1s2) / np.sqrt(1 - k1s2 * k1s2.conjugate())

        s1s3 = overlap(s1, s3, self._xs)
        s4 = lambda omega: (s3(omega) - s1(omega) * s1s3) / np.sqrt(1 - s1s3 * s1s3.conjugate())

        k4s4 = overlap(k4, s4, self._xs)

        """ Getting all coefficients """

        "First mode"

        A1 = zeta1 * uf1.conjugate()
        B1 = xi1 * ug1
        C1 = zeta1 * np.sqrt(1 - uf1.conjugate() * uf1) * k1h1.conjugate()
        D1 = xi1 * np.sqrt(1 - ug1.conjugate() * ug1)
        E1 = zeta1 * np.sqrt(1 - uf1.conjugate() * uf1) * np.sqrt(1 - k1h1.conjugate() * k1h1)

        #print('target1:', A1.conjugate() * A1 + 2 * B1.conjugate() * B1 + D1.conjugate() * D1)

        "second mode"

        A2 = zeta2 * uf2.conjugate()
        B2 = xi2 * ug2
        C2 = zeta2 * np.sqrt(1 - uf2.conjugate() * uf2) * (
                    k2h2.conjugate() * k1k2.conjugate() + np.sqrt(1 - k2h2.conjugate() * k2h2) * k1s2.conjugate())
        D2 = xi2 * np.sqrt(1 - ug2.conjugate() * ug2) * k1k2
        E2 = zeta2 * np.sqrt(1 - uf2.conjugate() * uf2) * (k2h2.conjugate() * np.sqrt(1 - k1k2.conjugate() * k1k2) * s1k3.conjugate()
                    + np.sqrt(1 - k2h2.conjugate() * k2h2) * np.sqrt(1 - k1s2.conjugate() * k1s2) * s1s3.conjugate())
        F2 = xi2 * np.sqrt(1 - ug2.conjugate() * ug2) * np.sqrt(1 - k1k2.conjugate() * k1k2) * s1k3
        G2 = zeta2 * np.sqrt(1 - uf2.conjugate() * uf2) * (k2h2.conjugate() * np.sqrt(1 - k1k2.conjugate() * k1k2) * np.sqrt(1 - s1k3.conjugate() * s1k3)
                    + np.sqrt(1 - k2h2.conjugate() * k2h2) * np.sqrt(1 - k1s2.conjugate() * k1s2) * np.sqrt(1 - s1s3.conjugate() * s1s3) * k4s4.conjugate())
        H2 = xi2 * np.sqrt(1 - ug2.conjugate() * ug2) * np.sqrt(1 - k1k2.conjugate() * k1k2) * np.sqrt(1 - s1k3.conjugate() * s1k3)

        norm = np.sqrt(A2 * A2.conjugate() - B2 * B2.conjugate() + C2 * C2.conjugate() - D2 * D2.conjugate() + E2 * E2.conjugate() - F2 * F2.conjugate() + G2 * G2.conjugate() - H2 * H2.conjugate())

        # Get the rotation angle for mixing with the final vacuum mode with the I2 coefficient
        theta_vac = np.arccos(norm)

        A2 = A2 / norm
        B2 = B2 / norm
        C2 = C2 / norm
        D2 = D2 / norm
        E2 = E2 / norm
        F2 = F2 / norm
        G2 = G2 / norm
        H2 = H2 / norm

        #print('target2:', A2.conjugate() * A2 + 2 * B2.conjugate() * B2 + D2.conjugate() * D2 + F2.conjugate() * F2 + H2.conjugate() * H2)

        return [A1, C1, E1, B1, D1], [A2, C2, E2, G2, B2, D2, F2, H2], theta_vac

    def _get_transformation_matrix(self, coefs1: List[float], coefs2: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the E and F matrices by introducing two ancillary modes. The coefficients of the ancillary modes are
        not unique, so one solution out of many is found.
        :param coefs1: The coefficients for the actual output mode 1 given in eq. 8 in the main paper
        :param coefs2: The coefficients for the actual output mode 2 given in eq. 8 in the main paper, and further
        decomposed to be orthogonal to mode 1.
        :return: Matrices E and F from G. Cariolaro and G. Pierobon (2016)
        """
        A1, C1, E1, B1, D1 = coefs1
        A2, C2, E2, G2, B2, D2, F2, H2 = coefs2

        row1 = np.array([-G2, H2, -E2, F2, -C2, D2, -A2, B2])
        row2 = np.array([-H2, G2, -F2, E2, -D2, C2, -B2, A2]).conjugate()
        row3 = np.array([0, 0, -E1, 0, -C1, D1, -A1, B1])
        row4 = np.array([0, 0, 0, E1, -D1, C1, -B1, A1]).conjugate()

        matrix = np.array([row1, row2, row3, row4])

        # Perform rref on the set of linear equations
        matrix = rref(matrix)

        # Mode 3
        A3 = 1
        B3 = 0
        C3 = 1
        D3 = 0
        E3 = - (matrix[3, 7] * A3 + matrix[3, 6] * B3 + matrix[3, 5] * C3 + matrix[3, 4] * D3)
        F3 = - (matrix[2, 7] * A3 + matrix[2, 6] * B3 + matrix[2, 5] * C3 + matrix[2, 4] * D3 + matrix[2, 3] * E3)
        G3 = - (matrix[1, 7] * A3 + matrix[1, 6] * B3 + matrix[1, 5] * C3 + matrix[1, 4] * D3 + matrix[1, 3] * E3 +
                matrix[1, 2] * F3)
        H3 = - (matrix[0, 7] * A3 + matrix[0, 6] * B3 + matrix[0, 5] * C3 + matrix[0, 4] * D3 + matrix[0, 3] * E3 +
                matrix[0, 2] * F3 + matrix[0, 1] * G3)

        norm = np.sqrt(A3.conjugate() * A3 - B3.conjugate() * B3 + C3.conjugate() * C3 - D3.conjugate() * D3 + E3.conjugate() * E3 - F3.conjugate() * F3 + G3 * G3.conjugate() - H3 * H3.conjugate())

        if np.imag(norm) != 0:
            A3 = 0
            B3 = 1
            C3 = 0
            D3 = 1
            E3 = - (matrix[3, 7] * A3 + matrix[3, 6] * B3 + matrix[3, 5] * C3 + matrix[3, 4] * D3)
            F3 = - (matrix[2, 7] * A3 + matrix[2, 6] * B3 + matrix[2, 5] * C3 + matrix[2, 4] * D3 + matrix[2, 3] * E3)
            G3 = - (matrix[1, 7] * A3 + matrix[1, 6] * B3 + matrix[1, 5] * C3 + matrix[1, 4] * D3 + matrix[1, 3] * E3 +
                    matrix[1, 2] * F3)
            H3 = - (matrix[0, 7] * A3 + matrix[0, 6] * B3 + matrix[0, 5] * C3 + matrix[0, 4] * D3 + matrix[0, 3] * E3 +
                    matrix[0, 2] * F3 + matrix[0, 1] * G3)

            norm = np.sqrt(A3.conjugate() * A3 - B3.conjugate() * B3 + C3.conjugate() * C3 - D3.conjugate() * D3 + E3.conjugate() * E3 - F3.conjugate() * F3 + G3 * G3.conjugate() - H3 * H3.conjugate())

        A3 = A3 / norm
        B3 = B3 / norm
        C3 = C3 / norm
        D3 = D3 / norm
        E3 = E3 / norm
        F3 = F3 / norm
        G3 = G3 / norm
        H3 = H3 / norm

        # Mode 4:
        row1 = np.array([-G3, H3, -E3, F3, -C3, D3, -A3, B3])
        row2 = np.array([-H3, G3, -F3, E3, -D3, C3, -B3, A3]).conjugate()
        row3 = np.array([0, 0, -E1, 0, -C1, D1, -A1, B1])
        row4 = np.array([0, 0, 0, E1, -D1, C1, -B1, A1]).conjugate()
        row5 = np.array([-G2, H2, -E2, F2, -C2, D2, -A2, B2])
        row6 = np.array([-H2, G2, -F2, E2, -D2, C2, -B2, A2]).conjugate()

        matrix = np.array([row1, row2, row3, row4, row5, row6])

        # Perform rref on the set of linear equations
        matrix = rref(matrix)

        # Get the coefficients for auxillary mode
        A4 = 1
        B4 = 0
        C4 = - (matrix[5, 7] * A4 + matrix[5, 6] * B4)
        D4 = - (matrix[4, 7] * A4 + matrix[4, 6] * B4 + matrix[4, 5] * C4)
        E4 = - (matrix[3, 7] * A4 + matrix[3, 6] * B4 + matrix[3, 5] * C4 + matrix[3, 4] * D4)
        F4 = - (matrix[2, 7] * A4 + matrix[2, 6] * B4 + matrix[2, 5] * C4 + matrix[2, 4] * D4 + matrix[2, 3] * E4)
        G4 = - (matrix[1, 7] * A4 + matrix[1, 6] * B4 + matrix[1, 5] * C4 + matrix[1, 4] * D4 + matrix[1, 3] * E4 +
                matrix[1, 2] * F4)
        H4 = - (matrix[0, 7] * A4 + matrix[0, 6] * B4 + matrix[0, 5] * C4 + matrix[0, 4] * D4 + matrix[0, 3] * E4 +
                matrix[0, 2] * F4 + matrix[0, 1] * G4)

        norm = np.sqrt(A4.conjugate() * A4 - B4.conjugate() * B4 + C4.conjugate() * C4 - D4.conjugate() * D4 + E4.conjugate() * E4 - F4.conjugate() * F4 + G4 * G4.conjugate() - H4 * H4.conjugate())

        if np.imag(norm) != 0:
            A4 = 0
            B4 = 1
            C4 = - (matrix[5, 7] * A4 + matrix[5, 6] * B4)
            D4 = - (matrix[4, 7] * A4 + matrix[4, 6] * B4 + matrix[4, 5] * C4)
            E4 = - (matrix[3, 7] * A4 + matrix[3, 6] * B4 + matrix[3, 5] * C4 + matrix[3, 4] * D4)
            F4 = - (matrix[2, 7] * A4 + matrix[2, 6] * B4 + matrix[2, 5] * C4 + matrix[2, 4] * D4 + matrix[2, 3] * E4)
            G4 = - (matrix[1, 7] * A4 + matrix[1, 6] * B4 + matrix[1, 5] * C4 + matrix[1, 4] * D4 + matrix[1, 3] * E4 +
                    matrix[1, 2] * F4)
            H4 = - (matrix[0, 7] * A4 + matrix[0, 6] * B4 + matrix[0, 5] * C4 + matrix[0, 4] * D4 + matrix[0, 3] * E4 +
                    matrix[0, 2] * F4 + matrix[0, 1] * G4)

            norm = np.sqrt(
                A4.conjugate() * A4 - B4.conjugate() * B4 + C4.conjugate() * C4 - D4.conjugate() * D4 + E4.conjugate() * E4 - F4.conjugate() * F4 + G4 * G4.conjugate() - H4 * H4.conjugate())

        A4 = A4 / norm
        B4 = B4 / norm
        C4 = C4 / norm
        D4 = D4 / norm
        E4 = E4 / norm
        F4 = F4 / norm
        G4 = G4 / norm
        H4 = H4 / norm

        # Write up the transformation matrices for output modes
        E = np.array([[A1, C1, E1, 0],
                      [A2, C2, E2, G2],
                      [A3, C3, E3, G3],
                      [A4, C4, E4, G4]], dtype=np.complex_)

        F = np.array([[B1, D1, 0, 0],
                      [B2, D2, F2, H2],
                      [B3, D3, F3, H3],
                      [B4, D4, F4, H4]], dtype=np.complex_)

        assert np.isclose(E @ E.conjugate().T - F @ F.conjugate().T, np.identity(4), atol=self._atol, rtol=self._rtol).all()
        assert np.isclose(E @ F.T, F @ E.T, atol=self._atol, rtol=self._rtol).all()
        return E, F

    def _get_parameters(self, U: np.ndarray, lambda_E: np.ndarray,
                        W_E: np.ndarray) -> Tuple[Tuple[float, float, float, float],
                                                  Tuple[List[float], List[float]],
                                                  Tuple[List[float], List[float]]]:
        """
        Finds the parameters for the rotation and squeezing transformations, as described in the main paper eq. 9
        :param U: The first rotation matrix from G. Cariolaro and G. Pierobon (2016) eq. 34.
        :param lambda_E: The squeezing matrix from G. Cariolaro and G. Pierobon (2016) eq. 34.
        :param W_E: The second rotation matrix from G. Cariolaro and G. Pierobon (2016) eq. 34.
        :return: The parameters for the rotation operators and squeezing operators in the order
        (squeezings), (rotations), (phases)
        """
        # Squeezing parameters
        r1 = np.arccosh(lambda_E[0, 0])
        r2 = np.arccosh(lambda_E[1, 1])
        r3 = np.arccosh(lambda_E[2, 2])
        r4 = np.arccosh(lambda_E[3, 3])

        theta3 = np.arcsin(np.abs(U[0, 3]))
        theta2 = np.arcsin(np.abs(U[0, 2] / np.cos(theta3)))
        theta1 = np.arccos(np.abs(U[0, 0] / (np.cos(theta2) * np.cos(theta3))))
        theta1s = [theta1, theta2, theta3]
        phi3 = phase(1j * U[0, 3])
        phi2 = phase(1j * U[0, 2])
        phi1 = phase(1j * U[0, 1]) - phase(U[0, 0])
        phi1s = [phi1, phi2, phi3]
        theta2s, phi2s = decompose_W_e(W_E.conjugate().T)
        return (r1, r2, r3, r4), (theta1s, theta2s), (phi1s, phi2s)

    def _transform_state(self, r1, r2, r3, r4, theta1s, theta2s, phi1s, phi2s, theta_vac) -> qt.Qobj:
        """
        Transforms the input quantum state to the output quantum state using the parameters from the Bloch-Messiah
        decomposition.
        :param r1: Squeezing parameter for mode 1
        :param r2: Squeezing parameter for mode 2
        :param theta1s: Rotation parameters for first rotation
        :param theta2s: Rotation parameters for second rotation
        :param phi1s: Phase parameters for first rotation
        :param phi2s: Phase parameters for second rotation
        :param theta_vac: Rotation angle for mixing with vacuum mode in the end
        :return: The output quantum state as a qutip state
        """
        # Define quantum system
        a = qt.destroy(self._N)
        I = qt.qeye(self._N)

        au = qt.tensor(a, I, I, I)
        a1 = qt.tensor(I, a, I, I)
        a2 = qt.tensor(I, I, a, I)
        a3 = qt.tensor(I, I, I, a)

        rho_u = self._psi0
        rho_1 = qt.basis(self._N, 0)
        rho_2 = qt.basis(self._N, 0)
        rho_3 = qt.basis(self._N, 0)

        rho = qt.ket2dm(qt.tensor(rho_u, rho_1, rho_2, rho_3))

        U_au_a1 = create_bs_interaction(au, a1, theta1s[0], phi1s[0])
        U_au_a2 = create_bs_interaction(au, a2, theta1s[1], phi1s[1])
        U_au_a3 = create_bs_interaction(au, a3, theta1s[2], phi1s[2])

        rho_t = U_au_a1 * U_au_a2 * U_au_a3 * rho * U_au_a3.dag() * U_au_a2.dag() * U_au_a1.dag()

        S1: qt.Qobj = qt.tensor(qt.squeeze(self._N, -r1), I, I, I)
        S2: qt.Qobj = qt.tensor(I, qt.squeeze(self._N, -r2), I, I)
        S3: qt.Qobj = qt.tensor(I, I, qt.squeeze(self._N, -r3), I)
        S4: qt.Qobj = qt.tensor(I, I, I, qt.squeeze(self._N, -r4))

        rho_t = S4 * S3 * S2 * S1 * rho_t * S1.dag() * S2.dag() * S3.dag() * S4.dag()

        V_au_a1 = create_bs_interaction(au, a1, theta2s[0], phi2s[0])
        V_au_a2 = create_bs_interaction(au, a2, theta2s[1], phi2s[1])
        V_au_a3 = create_bs_interaction(au, a3, theta2s[2], phi2s[2])
        V_a1_a2 = create_bs_interaction(a1, a2, theta2s[3], phi2s[3])
        V_a1_a3 = create_bs_interaction(a1, a3, theta2s[4], phi2s[4])

        rho_t = V_a1_a3 * V_a1_a2 * V_au_a3 * V_au_a2 * V_au_a1 * rho_t * V_au_a1.dag() * V_au_a2.dag() * V_au_a3.dag() * V_a1_a2.dag() * V_a1_a3.dag()

        rhov1v2 = rho_t.ptrace([0, 1])

        av2 = qt.tensor(I, a, I)
        a4 = qt.tensor(I, I, a)

        rho4 = qt.ket2dm(qt.basis(self._N, 0))

        rhov1v24 = qt.tensor(rhov1v2, rho4)

        V_a1_a4 = create_bs_interaction(av2, a4, theta_vac, 0)

        rhov1v2 = (V_a1_a4 * rhov1v24 * V_a1_a4.dag()).ptrace([0, 1])

        print(qt.expect(qt.tensor(a, I).dag() * qt.tensor(a, I), rhov1v2))
        print(qt.expect(qt.tensor(I, a).dag() * qt.tensor(I, a), rhov1v2))
        return rhov1v2
