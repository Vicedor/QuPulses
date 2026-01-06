import numpy as np
import qutip as qt
from thewalrus import quantum as twq
from scipy.integrate import quad, trapezoid
from scipy.interpolate import CubicSpline

from typing import List, Callable, Union, Tuple


def main():
    pass


def squeezed_fock_state():
    gamma = 1
    xi = 0.1

    n = 1
    tp = 4
    tau = 1
    N = 1
    Delta = 0

    M = 50

    omegas = np.linspace(-6, 6, 1000)
    u = lambda omega: np.sqrt(tau) / np.pi ** (1 / 4) * np.exp(-tau ** 2 / 2 * omega ** 2 + 1j * tp * omega)

    chi_X = lambda omega, xi: - ((gamma / 2 + xi) ** 2 + omega ** 2) / ((gamma / 2 - 1j * omega) ** 2 - xi ** 2)
    chi_P = lambda omega, xi: - ((gamma / 2 - xi) ** 2 + omega ** 2) / ((gamma / 2 - 1j * omega) ** 2 - xi ** 2)

    gamma_N_plus = lambda omega: (chi_X(omega, xi) ** N + chi_P(omega, xi) ** N) / 2
    gamma_N_minus = lambda omega: (chi_X(omega, xi) ** N - chi_P(omega, xi) ** N) / 2

    F = lambda omega: gamma_N_plus(omega - Delta)
    G = lambda omega: gamma_N_minus(omega - Delta).conjugate()

    def f(au: qt.Qobj, audag: qt.Qobj):
        return audag

    ss = SqueezingSystem(f, [M, M], u, F, G, omegas)


def overlap(f, g, xs):
    return quad(lambda omega: np.conjugate(f(omega)) * g(omega), xs[0], xs[-1], complex_func=True)[0]


class SqueezingSystem:
    def __init__(
            self,
            f: Callable[[qt.Qobj, qt.Qobj], qt.Qobj],
            dims: int | List[int],
            u: Callable[[float | np.ndarray], complex | np.ndarray],
            F: Union[Callable[[float | np.ndarray], complex | np.ndarray],
                     Callable[[float | np.ndarray, float | np.ndarray], complex | np.ndarray]],
            G: Union[Callable[[float | np.ndarray], complex | np.ndarray],
                     Callable[[float | np.ndarray, float | np.ndarray], complex | np.ndarray]],
            freq: np.ndarray,
    ):
        code = F.__code__
        self.no_args = code.co_argcount
        if self.no_args not in [1, 2]:
            raise Exception('F and G must take 1 or 2 arguments')
        self.u = u
        self.F = F
        self.G = G
        self.freq = freq
        self.fu, self.gu, self.zeta_u, self.xi_u = self.__compute_fu_gu()
        if len(dims) not in [1, 2]:
            raise Exception(f'dims must be length 1 or 2.')
        if isinstance(dims, int):
            dims = [dims]
        self.dims = dims
        self.f = f
        self.alpha, self.alpha_bar, self.beta_bar = self.__get_expectation_values()
        self.is_single_mode = self.__check_output_single_mode()
        if not self.is_single_mode:
            if len(self.dims) is not 2:
                raise Exception(f'dims must be length 2 when input state does not obey single mode condition (eq. 9)')
        self.vs = self.__compute_output_modes()

    def __compute_fu_gu(self):
        if self.no_args is 1:
            fu_temp = lambda omega: self.F(omega) * self.u(omega)
            gu_temp = lambda omega: np.conjugate(self.G(omega)) * np.conjugate(self.u(omega))
        else:
            # TODO: check axis is correct
            fu_array = trapezoid(self.F(self.freq, self.freq) @ self.u(self.freq), self.freq, axis=1)
            gu_array = trapezoid(np.conjugate(self.G(self.freq, self.freq)) @ np.conjugate(self.u(self.freq)),
                                 self.freq, axis=1)
            fu_temp = CubicSpline(self.freq, fu_array)
            gu_temp = CubicSpline(self.freq, gu_array)

        zeta_u = np.sqrt(overlap(fu_temp, fu_temp, self.freq))
        xi_u = np.sqrt(overlap(gu_temp, gu_temp, self.freq))

        fu = lambda omega: fu_temp(omega) / zeta_u
        gu = lambda omega: gu_temp(omega) / xi_u

        return fu, gu, zeta_u, xi_u

    def __compute_fvs_gvs(self):
        if self.is_single_mode:
            v = self.vs[0]
            if self.no_args is 1:
                fv_temp = lambda omega: np.conjugate(self.F(omega)) * v(omega)
                gv_temp = lambda omega: self.G(omega) * np.conjugate(v(omega))
            else:
                # TODO: check axis is correct
                fv_array = trapezoid(np.conjugate(self.F(self.freq, self.freq)) @ v(self.freq), self.freq, axis=1)
                gv_array = trapezoid(self.G(self.freq, self.freq) @ np.conjugate(v(self.freq)), self.freq, axis=1)
                fv_temp = CubicSpline(self.freq, fv_array)
                gv_temp = CubicSpline(self.freq, gv_array)
            zeta_v = np.sqrt(overlap(fv_temp, fv_temp, self.freq))
            xi_v = np.sqrt(overlap(gv_temp, gv_temp, self.freq))

            fv = lambda omega: fv_temp(omega) / zeta_v
            gv = lambda omega: gv_temp(omega) / xi_v

            return fv, gv, zeta_v, xi_v

        else:
            v1, v2 = self.vs
            if self.no_args is 1:
                fv1_temp = lambda omega: np.conjugate(self.F(omega)) * v1(omega)
                gv1_temp = lambda omega: self.G(omega) * np.conjugate(v1(omega))
                fv2_temp = lambda omega: np.conjugate(self.F(omega)) * v2(omega)
                gv2_temp = lambda omega: self.G(omega) * np.conjugate(v2(omega))
            else:
                # TODO: check axis is correct
                fv1_array = trapezoid(np.conjugate(self.F(self.freq, self.freq)) @ v1(self.freq), self.freq, axis=1)
                gv1_array = trapezoid(self.G(self.freq, self.freq) @ np.conjugate(v1(self.freq)), self.freq, axis=1)
                fv1_temp = CubicSpline(self.freq, fv1_array)
                gv1_temp = CubicSpline(self.freq, gv1_array)
                fv2_array = trapezoid(np.conjugate(self.F(self.freq, self.freq)) @ v2(self.freq), self.freq, axis=1)
                gv2_array = trapezoid(self.G(self.freq, self.freq) @ np.conjugate(v2(self.freq)), self.freq, axis=1)
                fv2_temp = CubicSpline(self.freq, fv2_array)
                gv2_temp = CubicSpline(self.freq, gv2_array)
            zeta_v1 = np.sqrt(overlap(fv1_temp, fv1_temp, self.freq))
            xi_v1 = np.sqrt(overlap(gv1_temp, gv1_temp, self.freq))
            zeta_v2 = np.sqrt(overlap(fv2_temp, fv2_temp, self.freq))
            xi_v2 = np.sqrt(overlap(gv2_temp, gv2_temp, self.freq))

            fv1 = lambda omega: fv1_temp(omega) / zeta_v1
            gv1 = lambda omega: gv1_temp(omega) / xi_v1
            fv2 = lambda omega: fv2_temp(omega) / zeta_v2
            gv2 = lambda omega: gv2_temp(omega) / xi_v2

            return fv1, gv1, zeta_v1, xi_v1, fv2, gv2, zeta_v2, xi_v2

    def __get_expectation_values(self):
        N = self.dims[0]
        vac = qt.basis(N, 0)
        a = qt.destroy(N)
        psi = self.f(a, a.dag()) @ vac
        adaga = qt.expect(a.dag() @ a, psi)
        aa = qt.expect(a @ a, psi)
        a = qt.expect(a, psi)
        return a, adaga, aa

    def __compute_output_modes(self):
        gu_fu = overlap(self.gu, self.fu, self.freq)
        if self.is_single_mode:
            k = np.sqrt(self.alpha * np.conjugate(self.alpha) * (self.zeta_u ** 2 + self.xi_u ** 2)
                        + 2 * self.zeta_u * self.xi_u * np.real(self.alpha ** 2 * gu_fu))
            v = lambda omega: (self.alpha * self.zeta_u * self.fu(omega)
                               + np.conjugate(self.alpha) * self.xi_u * self.gu(omega)) / k
            v_fu = overlap(v, self.fu, self.freq)
            h = lambda omega: (self.fu(omega) - v_fu * v(omega)) / np.sqrt(1 - v_fu * np.conjugate(v_fu))
            return [v, h]
        else:
            gamma = gu_fu
            delta = np.sqrt(1 - gamma * np.conjugate(gamma))
            hu = lambda omega: (self.gu(omega) - gamma * self.fu(omega)) / delta
            x = (self.alpha_bar * self.zeta_u ** 2
                 + 2 * self.zeta_u * self.xi_u * np.real(np.conjugate(self.beta_bar) * gamma)
                 + self.alpha_bar * self.xi_u ** 2 * gamma * np.conjugate(gamma))
            y = self.beta_bar * self.zeta_u * self.xi_u * delta + self.alpha_bar * self.zeta_u ** 2 * gamma * delta
            z = self.alpha_bar * self.xi_u ** 2 * delta ** 2

            c1_fu = (x - z + np.sqrt((x - z) ** 2 + 4 * y * np.conjugate(y)))
            c1_hu = 2 * np.conjugate(y)
            v1_norm = np.sqrt(2 * ((x - z) ** 2 + 4 * y * np.conjugate(y)
                                   + (x - z) * np.sqrt((x - z) ** 2 + 4 * y * np.conjugate(y))))
            v1 = lambda omega: (c1_fu * self.fu(omega) + c1_hu * hu(omega)) / v1_norm

            c2_fu = (x - z - np.sqrt((x - z) ** 2 + 4 * y * np.conjugate(y)))
            c2_hu = 2 * np.conjugate(y)
            v2_norm = np.sqrt(2 * ((x - z) ** 2 + 4 * y * np.conjugate(y)
                                   - (x - z) * np.sqrt((x - z) ** 2 + 4 * y * np.conjugate(y))))
            v2 = lambda omega: (c2_fu * self.fu(omega) + c2_hu * hu(omega)) / v2_norm
            return [v1, v2]

    def get_squeezed_output_state(self) -> qt.Qobj:
        au: qt.Qobj = self.__decompose_input_operator()
        squeezed_vacuum: qt.Qobj = self.__get_squeezed_vacuum_state()
        return self.f(au, au.dag()) @ squeezed_vacuum @ self.f(au.dag(), au)

    def __decompose_input_operator(self) -> qt.Qobj:
        if self.is_single_mode:
            v, h = self.vs
            fu_v = overlap(self.fu, v, self.freq)
            v_gu = overlap(v, self.gu, self.freq)
            fu_h = overlap(self.fu, h, self.freq)
            h_gu = overlap(h, self.gu, self.freq)
            if len(self.dims) is 1:
                dims = self.dims + [0]
            else:
                dims = self.dims
            N1, N2 = dims
            if N2 is not 0:
                bv: qt.Qobj = qt.tensor(qt.destroy(N1), qt.qeye(N2))
                bh: qt.Qobj = qt.tensor(qt.qeye(N1), qt.destroy(N2))
                au = (self.zeta_u * fu_v * bv - self.xi_u * v_gu * bv.dag()
                      + self.zeta_u * fu_h * bh - self.xi_u * h_gu * bh.dag())
            else:
                bv: qt.Qobj = qt.destroy(N1)
                au = self.zeta_u * fu_v * bv - self.xi_u * v_gu * bv.dag()
        else:
            v1, v2 = self.vs
            fu_v1 = overlap(self.fu, v1, self.freq)
            v1_gu = overlap(v1, self.gu, self.freq)
            fu_v2 = overlap(self.fu, v2, self.freq)
            v2_gu = overlap(v2, self.gu, self.freq)
            N1, N2 = self.dims
            bv1 = qt.tensor(qt.destroy(N1), qt.qeye(N2))
            bv2 = qt.tensor(qt.qeye(N1), qt.destroy(N2))
            au = self.zeta_u * fu_v1 * bv1 - self.xi_u * v1_gu * bv1.dag() + self.zeta_u * fu_v2 * bv2 - self.xi_u * v2_gu * bv2.dag()
        return au

    def __check_output_single_mode(self) -> bool:

        return np.isclose(self.alpha_bar ** 2, self.beta_bar * np.conjugate(self.beta_bar))

    def __get_squeezed_vacuum_state(self) -> qt.Qobj:
        E, F = self.__get_E_and_F()
        cov = get_covariance_matrix(E, F)
        if len(self.dims) == 1:
            density_matrix = qt.Qobj(twq.density_matrix(mu=np.array([0, 0]), cov=cov / 2,
                                                        cutoff=self.dims, normalize=True, hbar=1))
        else:
            density_matrix = twq.density_matrix(mu=np.array([0, 0, 0, 0]), cov=cov / 2, cutoff=self.dims, hbar=1)
            density_matrix = convert_density_matrix(density_matrix, self.dims)
        return density_matrix

    def __get_E_and_F(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_single_mode:
            fv, gv, zeta_v, xi_v = self.__compute_fvs_gvs()
            fv_u = overlap(fv, self.u, self.freq)
            u_gv = overlap(self.u, gv, self.freq)
            E = np.array([[zeta_v * fv_u, zeta_v * np.sqrt(1 - fv_u * np.conjugate(fv_u))]])
            F = np.array([[xi_v * u_gv, xi_v * np.sqrt(1 - fv_u * np.conjugate(fv_u))]])
        else:
            fv1, gv1, zeta_v1, xi_v1, fv2, gv2, zeta_v2, xi_v2 = self.__compute_fvs_gvs()
            fv1_u = overlap(fv1, self.u, self.freq)
            t = lambda omega: ((fv1(omega) - np.conjugate(fv1_u) * self.u(omega))
                               / np.sqrt(1 - fv1_u * np.conjugate(fv1_u)))
            fv2_u = overlap(fv2, self.u, self.freq)
            fv2_t = overlap(fv2, t, self.freq)
            s = lambda omega: ((fv2(omega) - np.conjugate(fv2_u) * self.u(omega) - np.conjugate(fv2_t) * t(omega))
                               / np.sqrt(1 - fv2_u * np.conjugate(fv2_u) - fv2_t * np.conjugate(fv2_t)))

            fv1_modes = np.array([fv1_u, overlap(fv1, t, self.freq), overlap(fv1, s, self.freq)])
            fv2_modes = np.array([fv2_u, fv2_t, overlap(fv2, s, self.freq)])

            modes_gv1 = np.array([overlap(self.u, gv1, self.freq), overlap(t, gv1, self.freq),
                                  overlap(s, gv1, self.freq)])
            modes_gv2 = np.array([overlap(self.u, gv2, self.freq), overlap(t, gv2, self.freq),
                                  overlap(s, gv2, self.freq)])

            E = np.block([[zeta_v1 * fv1_modes], [zeta_v2 * fv2_modes]])
            F = np.block([[xi_v1 * modes_gv1], [xi_v2 * modes_gv2]])
        return E, F


def get_covariance_matrix(
        E: np.ndarray,
        F: np.ndarray
) -> np.ndarray:
    dims = E.shape
    dims_F = F.shape
    if not dims == dims_F:
        raise Exception('Dimension of E and F must be the same')

    N = dims[0]
    Identity = np.eye(N)

    EFT = E @ F.T
    FFdag = F @ F.conjugate().T

    cov_2 = np.block([[np.real(EFT + FFdag), np.imag(EFT - FFdag)],
                      [np.imag(EFT + FFdag), np.real(FFdag - EFT)]])

    return Identity + 2 * cov_2


def convert_density_matrix(density_matrix, dims):
    # TODO: vectorize!
    N1, N2 = dims
    new_density_matrix = np.zeros([N1 * N2, N1 * N2], dtype=np.complex128)
    for n1 in range(N1):
        for m1 in range(N1):
            for n2 in range(N2):
                for m2 in range(N2):
                    rhon1n2m1m2 = density_matrix[n1][m1][n2][m2]
                    new_density_matrix[n1 * N1 + n2, m1 * N1 + m2] = rhon1n2m1m2
    return qt.Qobj(new_density_matrix, dims=[[N1, N2], [N1, N2]])


if __name__ == '__main__':
    main()
