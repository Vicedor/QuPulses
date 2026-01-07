import numpy as np
import qutip as qt
from thewalrus import quantum as twq
from scipy.integrate import quad
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import List, Callable, Tuple


def main():
    squeezed_cat_state()
    squeezed_fock_state()


def squeezed_cat_state():
    gamma = 1
    xi = 0.1

    alpha = 1 + 1j

    tp = 4
    tau = 1
    N = 1
    Delta = 0

    M = 80

    # Array of frequencies
    omegas = np.linspace(-4, 4, 1000)

    # A gaussian input pulse in frequency domain (fourier transform of time domain)
    u = lambda omega: np.sqrt(tau) / np.pi ** (1 / 4) * np.exp(-tau ** 2 / 2 * omega ** 2 + 1j * tp * omega)

    # The functions needed for a TWPA transformation of input pulse
    chi_X = lambda omega, xi: - ((gamma / 2 + xi) ** 2 + omega ** 2) / ((gamma / 2 - 1j * omega) ** 2 - xi ** 2)
    chi_P = lambda omega, xi: - ((gamma / 2 - xi) ** 2 + omega ** 2) / ((gamma / 2 - 1j * omega) ** 2 - xi ** 2)

    gamma_N_plus = lambda omega: (chi_X(omega, xi) ** N + chi_P(omega, xi) ** N) / 2
    gamma_N_minus = lambda omega: (chi_X(omega, xi) ** N - chi_P(omega, xi) ** N) / 2

    F = lambda omega: gamma_N_plus(omega - Delta)
    G = lambda omega: gamma_N_minus(omega - Delta).conjugate()

    def f(au: qt.Qobj, audag: qt.Qobj):
        D = (alpha * audag - np.conjugate(alpha) * au).expm()
        return (D + 1j * D.dag()) / np.sqrt(2)

    fu_temp = lambda omega: F(omega) * u(omega)
    gu_temp = lambda omega: G(2*Delta - omega) * u(2*Delta - omega).conjugate()

    zeta_u = np.sqrt(overlap(fu_temp, fu_temp, omegas))
    xi_u = np.sqrt(overlap(gu_temp, gu_temp, omegas))

    fu = lambda omega: fu_temp(omega) / zeta_u
    gu = lambda omega: gu_temp(omega) / xi_u

    ss = SqueezingSystem(f, M, u, fu, gu, zeta_u, xi_u, omegas)

    v = ss.get_output_modes()[0]

    fv_temp = lambda omega: F(omega).conjugate() * v(omega)
    gv_temp = lambda omega: G(omega) * v(2 * Delta - omega).conjugate()

    zeta_v = np.sqrt(overlap(fv_temp, fv_temp, omegas))
    xi_v = np.sqrt(overlap(gv_temp, gv_temp, omegas))

    fv = lambda omega: fv_temp(omega) / zeta_v
    gv = lambda omega: gv_temp(omega) / xi_v

    rho_squeezed_cat_state = ss.get_squeezed_output_state(v, fv, gv, zeta_v, xi_v)

    # Find the Wigner function numerically
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(rho_squeezed_cat_state, xvec, xvec)

    # normalize colors to the length of data
    nrm = mpl.colors.Normalize(w.min(), w.max())

    # Plot the Wigner function
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()


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

    fu_temp = lambda omega: F(omega) * u(omega)
    gu_temp = lambda omega: G(2*Delta - omega) * u(2*Delta - omega).conjugate()

    zeta_u = np.sqrt(overlap(fu_temp, fu_temp, omegas))
    xi_u = np.sqrt(overlap(gu_temp, gu_temp, omegas))

    fu = lambda omega: fu_temp(omega) / zeta_u
    gu = lambda omega: gu_temp(omega) / xi_u

    ss = SqueezingSystem(f, [M, M], u, fu, gu, zeta_u, xi_u, omegas)

    v1, v2 = ss.get_output_modes()

    fv1_temp = lambda omega: F(omega).conjugate() * v1(omega)
    gv1_temp = lambda omega: G(omega) * v1(2*Delta - omega).conjugate()
    fv2_temp = lambda omega: F(omega).conjugate() * v2(omega)
    gv2_temp = lambda omega: G(omega) * v2(2*Delta - omega).conjugate()

    zeta_v1 = np.sqrt(overlap(fv1_temp, fv1_temp, omegas))
    xi_v1 = np.sqrt(overlap(gv1_temp, gv1_temp, omegas))
    zeta_v2 = np.sqrt(overlap(fv2_temp, fv2_temp, omegas))
    xi_v2 = np.sqrt(overlap(gv2_temp, gv2_temp, omegas))

    fv1 = lambda omega: fv1_temp(omega) / zeta_v1
    gv1 = lambda omega: gv1_temp(omega) / xi_v1
    fv2 = lambda omega: fv2_temp(omega) / zeta_v2
    gv2 = lambda omega: gv2_temp(omega) / xi_v2

    rho_squeezed_fock_state = ss.get_squeezed_output_state([v1, v2], [fv1, fv2], [gv1, gv2],
                                                           [zeta_v1, zeta_v2], [xi_v1, xi_v2])

    # Find the Wigner function numerically
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(qt.ptrace(rho_squeezed_fock_state, 0), xvec, xvec)

    # normalize colors to the length of data
    nrm = mpl.colors.Normalize(w.min(), w.max())

    # Plot the Wigner function
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    # Find the Wigner function numerically
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(qt.ptrace(rho_squeezed_fock_state, 1), xvec, xvec)

    # normalize colors to the length of data
    nrm = mpl.colors.Normalize(w.min(), w.max())

    # Plot the Wigner function
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()


def overlap(f, g, xs):
    return quad(lambda omega: np.conjugate(f(omega)) * g(omega), xs[0], xs[-1], complex_func=True)[0]


class SqueezingSystem:
    def __init__(
            self,
            f: Callable[[qt.Qobj, qt.Qobj], qt.Qobj],
            dims: int | List[int],
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
        if isinstance(dims, int):
            dims = [dims]
        if len(dims) not in [1, 2]:
            raise Exception(f'dims must be length 1 or 2.')
        self.dims = dims
        self.f = f
        self.alpha, self.alpha_bar, self.beta_bar = self.__get_expectation_values()
        self.is_single_mode = self.__check_output_single_mode()
        if not self.is_single_mode:
            if len(self.dims) != 2:
                raise Exception(f'dims must be length 2 when input state does not obey single mode condition (eq. 9)')

    def __get_expectation_values(self):
        N = self.dims[0]
        vac = qt.basis(N, 0)
        a = qt.destroy(N)
        psi = self.f(a, a.dag()) @ vac
        adaga = qt.expect(a.dag() @ a, psi)
        aa = qt.expect(a @ a, psi)
        return np.sqrt(aa), adaga, aa

    def get_output_modes(self):
        gu_fu = overlap(self.gu, self.fu, self.freq)
        if self.is_single_mode:
            k = np.sqrt(self.alpha * np.conjugate(self.alpha) * (self.zeta_u ** 2 + self.xi_u ** 2)
                        + 2 * self.zeta_u * self.xi_u * np.real(self.alpha ** 2 * gu_fu))
            v = lambda omega: (self.alpha * self.zeta_u * self.fu(omega)
                               + np.conjugate(self.alpha) * self.xi_u * self.gu(omega)) / k
            return [v]
        else:
            gamma = gu_fu
            delta = np.sqrt(1 - gamma * np.conjugate(gamma))
            hu = lambda omega: (self.gu(omega) - gamma * self.fu(omega)) / delta
            x = (self.alpha_bar * self.zeta_u ** 2
                 + 2 * self.zeta_u * self.xi_u * np.real(np.conjugate(self.beta_bar) * gamma)
                 + self.alpha_bar * self.xi_u ** 2 * gamma * np.conjugate(gamma))
            y = self.beta_bar * self.zeta_u * self.xi_u * delta + self.alpha_bar * self.xi_u ** 2 * gamma * delta
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

    def get_squeezed_output_state(self, v, fv, gv, zeta_v, xi_v) -> qt.Qobj:
        au: qt.Qobj = self.decompose_input_operator(v)
        squeezed_vacuum: qt.Qobj = self.get_squeezed_vacuum_state(fv, gv, zeta_v, xi_v)
        return self.f(au, au.dag()) @ squeezed_vacuum @ self.f(au, au.dag()).dag()

    def decompose_input_operator(self, v) -> qt.Qobj:
        if self.is_single_mode:
            fu_v = overlap(self.fu, v, self.freq)
            v_gu = overlap(v, self.gu, self.freq)
            N = self.dims[0]
            bv: qt.Qobj = qt.destroy(N)
            au = self.zeta_u * fu_v * bv - self.xi_u * v_gu * bv.dag()
        else:
            v1, v2 = v
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

    def get_squeezed_vacuum_state(self, fv, gv, zeta_v, xi_v) -> qt.Qobj:
        E, F = self.get_E_and_F(fv, gv, zeta_v, xi_v)
        cov = get_covariance_matrix(E, F)
        if len(self.dims) == 1:
            density_matrix = qt.Qobj(twq.density_matrix(mu=np.array([0, 0]), cov=cov / 2,
                                                        cutoff=self.dims[0], normalize=True, hbar=1))
        else:
            density_matrix = twq.density_matrix(mu=np.array([0, 0, 0, 0]), cov=cov / 2, cutoff=self.dims[0], hbar=1)
            density_matrix = convert_density_matrix(density_matrix, self.dims)
        return density_matrix

    def get_E_and_F(self, fv, gv, zeta_v, xi_v) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_single_mode:
            fv_u = overlap(fv, self.u, self.freq)
            u_gv = overlap(self.u, gv, self.freq)
            E = np.array([[zeta_v * fv_u, zeta_v * np.sqrt(1 - fv_u * np.conjugate(fv_u))]])
            F = np.array([[xi_v * u_gv, zeta_v * np.sqrt(1 - fv_u * np.conjugate(fv_u))]])
        else:
            fv1, fv2 = fv
            gv1, gv2 = gv
            zeta_v1, zeta_v2 = zeta_v
            xi_v1, xi_v2 = xi_v
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
    Identity = np.eye(2 * N)

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
