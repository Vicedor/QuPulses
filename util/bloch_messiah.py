import numpy as np
import qutip as qt
from scipy.integrate import quad
from cmath import phase
from scipy.linalg import sqrtm
from typing import Callable, Tuple, List


def overlap(f, g, xs):
    return quad(lambda omega: np.conjugate(f(omega)) * g(omega), xs[0], xs[-1], complex_func=True)[0]


def create_bs_interaction(a: qt.Qobj, b: qt.Qobj, theta: float, phi: float) -> qt.Qobj:
    return (-1j * theta * (a.dag() * b * np.exp(1j * phi) + a * b.dag() * np.exp(-1j * phi))).expm(method='sparse')


class TwoModeBlochMessiah:
    def __init__(self, u: Callable[[float], float], f_temp: Callable[[float], float], g_temp: Callable[[float], float],
                 xs: np.ndarray, N: int, psi0: qt.Qobj):
        self._u: Callable[[float], float] = u
        self._f_temp: Callable[[float], float] = f_temp
        self._g_temp: Callable[[float], float] = g_temp
        self._xs: np.ndarray = xs
        self._N = N
        self._psi0 = psi0

    def get_output_state(self) -> qt.Qobj:
        coefs, theta_vac = self._get_mode_coefs()
        E_mat, F_mat = self._get_transformation_matrix(coefs)

        scriptU, lambda_E, scriptW_E = self.bloch_messiah(E_mat, F_mat)
        rs, thetas, phis = self._get_parameters(scriptU, lambda_E, scriptW_E)
        #r1, r2 = rs
        #theta1, theta2 = thetas
        #phi1, phi2 = phis

        #rhov = self.transform_state(r1, r2, theta1, theta2, phi1, phi2, theta_vac)
        rhov = self._transform_state(*rs, *thetas, *phis, theta_vac)
        return rhov

    def _get_mode_coefs(self) -> Tuple[List[float], float]:
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

        #print('target:', (A1 + B1).conjugate() * (A1 + B1) + B1.conjugate() * B1 + D1.conjugate() * D1)
        print('target:', A1.conjugate() * A1 + 2 * B1.conjugate() * B1 + D1.conjugate() * D1)

        norm = np.sqrt(A1.conjugate() * A1 - B1.conjugate() * B1 + C1.conjugate() * C1 - D1.conjugate() * D1)

        theta_vac = np.arccos(norm)

        A1 = A1 / norm
        B1 = B1 / norm
        C1 = C1 / norm
        D1 = D1 / norm

        return [A1, C1, B1, D1], theta_vac

    def _get_transformation_matrix(self, coefs: List[float]) -> Tuple[np.ndarray, np.ndarray]:
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

        #assert np.isclose(E @ E.conjugate().T - F @ F.conjugate().T, np.identity(2)).all()
        #assert np.isclose(E @ F.T, F @ E.T).all()
        return E, F

    def bloch_messiah(self, E: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param
        """
        # Step 1 and 2
        lambda_E_sq, U = np.linalg.eig(E @ E.conjugate().T)

        lambda_E = np.diag(np.sqrt(lambda_E_sq))
        lambda_F = np.sqrt(U.conjugate().T @ F @ F.conjugate().T @ U)

        Vh_E = np.linalg.inv(lambda_E) @ U.conjugate().T @ E
        W_E = Vh_E.conjugate().T
        W_F = (np.linalg.inv(lambda_F) @ U.conjugate().T @ F).conjugate().T

        assert np.isclose(U @ lambda_E @ W_E.conjugate().T, E).all()
        assert np.isclose(U @ lambda_F @ W_F.conjugate().T, F).all()

        # Step 3

        G = W_E.conjugate().T @ W_F.conjugate()

        D = sqrtm(G)

        # Step 5

        scriptW_E = (D.T @ W_F.T).conjugate().T
        scriptU = U @ D
        scriptW_F = W_F @ D

        #assert np.all(np.isclose(scriptU @ lambda_E @ scriptW_E.conjugate().T, E))
        #assert np.all(np.isclose(scriptU @ lambda_F @ scriptW_F.conjugate().T, F))
        #assert np.all(np.isclose(scriptW_F, scriptW_E.conjugate()))
        return scriptU, lambda_E, scriptW_E

    def _get_parameters(self, scriptU: np.ndarray, lambda_E: np.ndarray,
                       scriptW_E: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        # Squeezing parameters
        r1 = np.arccosh(lambda_E[0, 0])
        r2 = np.arccosh(lambda_E[1, 1])

        theta1 = np.arccos(np.abs(scriptW_E.conjugate().T[0, 0]))
        phi1 = phase(1j * scriptW_E.conjugate().T[0, 1]) - phase(scriptW_E.conjugate().T[1, 1])
        theta2 = np.arccos(np.abs(scriptU[0, 0]))
        phi2 = phase(1j * scriptU[0, 1]) - phase(scriptU[0, 0])
        return (r1, r2), (theta1, theta2), (phi1, phi2)

    def _transform_state(self, r1, r2, theta1, theta2, phi1, phi2, theta_vac) -> qt.Qobj:
        a = qt.destroy(self._N)
        I = qt.qeye(self._N)

        # Define quantum system
        au = qt.tensor(a, I)
        a1 = qt.tensor(I, a)

        rho_u = qt.ket2dm(self._psi0)
        rho_1 = qt.ket2dm(qt.basis(self._N, 0))
        rho_2 = qt.ket2dm(qt.basis(self._N, 0))

        rho = qt.tensor(rho_u, rho_1)

        U_au_a1 = create_bs_interaction(au, a1, theta1, phi1)

        rho_t = U_au_a1 * rho * U_au_a1.dag()

        S1: qt.Qobj = qt.tensor(qt.squeeze(self._N, -r1), I)
        S2: qt.Qobj = qt.tensor(I, qt.squeeze(self._N, -r2))

        rho_t = S2 * S1 * rho_t * S1.dag() * S2.dag()

        V_au_a1 = create_bs_interaction(au, a1, theta2, phi2)

        rho_t = V_au_a1 * rho_t * V_au_a1.dag()

        rhov = rho_t.ptrace(0)

        av = qt.tensor(a, I)
        a2 = qt.tensor(I, a)

        rhov2 = qt.tensor(rhov, rho_2)

        V_a1_a4 = create_bs_interaction(av, a2, theta_vac, 0)

        rhov = (V_a1_a4 * rhov2 * V_a1_a4.dag()).ptrace(0)
        print(qt.expect(a.dag() * a, rhov))
        return rhov

