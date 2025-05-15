#! python3
from typing import List, Tuple
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import qutip as qt
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from util.bloch_messiah import SingleModeBlochMessiah
from util import math_functions as m
from util import physics_functions as ph

from thewalrus import quantum as twq


gamma = 1
xi = 0.1

n = 1
alpha = 1
tp = 4
tau = 1
N = 1
Delta = 0

M = 50


omegas = np.linspace(-4, 4, 1000)
u_tilde = lambda omega: np.sqrt(tau) / np.pi**(1/4) * np.exp(-tau**2 / 2 * omega**2 + 1j * tp * omega)
u = lambda omega: m.normalized_hermite_polynomial(tp, tau, 0)(omega) * u_tilde(omega)


chi_X = lambda omega, xi: - ((gamma / 2 + xi)**2 + omega**2) / ((gamma/2 - 1j*omega)**2 - xi**2)
chi_P = lambda omega, xi: - ((gamma / 2 - xi)**2 + omega**2) / ((gamma/2 - 1j*omega)**2 - xi**2)

gamma_N_plus = lambda omega: (chi_X(omega, xi)**N + chi_P(omega, xi)**N) / 2
gamma_N_minus = lambda omega: (chi_X(omega, xi)**N - chi_P(omega, xi)**N) / 2

F = lambda omega: gamma_N_plus(omega - Delta)
G = lambda omega: gamma_N_minus(omega - Delta).conjugate()


def main():
    rho_bm = bloch_messiah_density_matrix()
    rho_ana = squeezed_vacuum_density_matrix()


def bloch_messiah_density_matrix():
    g1_list = np.zeros([len(omegas), len(omegas)], dtype=np.complex128)
    for i, omega1 in enumerate(omegas):
        for j, omega2 in enumerate(omegas):
            g1_list[i, j] = g1_fock(omega1, omega2)
        if (i + 1) % 10 == 0:
            print(i + 1)

    vals, vecs = ph.convert_autocorr_mat_to_vals_and_vecs(g1_list, omegas, n=2, trim=True)

    v1 = CubicSpline(omegas, -vecs[0].real + 1j * vecs[0].imag)
    v2 = CubicSpline(omegas, -vecs[1].real + 1j * vecs[1].imag)

    f1_temp = lambda omega: F(omega).conjugate() * v1(omega)
    g1_temp = lambda omega: G(omega) * v1(2*Delta - omega).conjugate()
    f2_temp = lambda omega: F(omega).conjugate() * v2(omega)
    g2_temp = lambda omega: G(omega) * v2(2*Delta - omega).conjugate()

    t0 = time.time()
    bm1 = SingleModeBlochMessiah(u, f1_temp, g1_temp, omegas, M, qt.basis(M, n))
    rhov1 = bm1.get_output_state()
    t1 = time.time()

    # Find the Wigner function numerically
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(rhov1, xvec, xvec)

    # normalize colors to the length of data
    nrm = mpl.colors.Normalize(w.min(), w.max())

    # Plot the Wigner function
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    t2 = time.time()
    bm2 = SingleModeBlochMessiah(u, f2_temp, g2_temp, omegas, M, qt.basis(M, n))
    rhov2 = bm2.get_output_state()
    t3 = time.time()
    print(f'Numerical single mode solution time={t3 - t2 + t1 - t0}')

    # Find the Wigner function numerically
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(rhov2, xvec, xvec)

    # normalize colors to the length of data
    nrm = mpl.colors.Normalize(w.min(), w.max())

    # Plot the Wigner function
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    a = qt.destroy(M)

    print('a1daga1:', qt.expect(a.dag() * a, rhov1))
    print('a2daga2:', qt.expect(a.dag() * a, rhov2))

    return rhov1, rhov2


def squeezed_vacuum_density_matrix():
    g1_list = np.zeros([len(omegas), len(omegas)], dtype=np.complex128)
    for i, omega1 in enumerate(omegas):
        for j, omega2 in enumerate(omegas):
            g1_list[i, j] = g1_fock(omega1, omega2)
        if (i + 1) % 10 == 0:
            print(i + 1)

    vals, vecs = ph.convert_autocorr_mat_to_vals_and_vecs(g1_list, omegas, n=2, trim=True)

    v1 = CubicSpline(omegas, -vecs[0].real + 1j * vecs[0].imag)
    v2 = CubicSpline(omegas, -vecs[1].real + 1j * vecs[1].imag)

    f1_temp = lambda omega: F(omega).conjugate() * v1(omega)
    g1_temp = lambda omega: G(omega) * v1(2*Delta - omega).conjugate()
    f2_temp = lambda omega: F(omega).conjugate() * v2(omega)
    g2_temp = lambda omega: G(omega) * v2(2*Delta - omega).conjugate()

    mode1_coefs, mode2_coefs = get_mode_coefs(u, f1_temp, g1_temp, f2_temp, g2_temp, omegas)

    A1, C1, E1, B1, D1 = mode1_coefs
    A2, C2, E2, G2, B2, D2, F2, H2 = mode2_coefs

    E_mat = np.array([[A1, C1, E1, 0],
                      [A2, C2, E2, G2]])
    F_mat = np.array([[B1, D1, 0, 0],
                      [B2, D2, F2, H2]])

    A11 = E_mat + np.conjugate(E_mat) + F_mat + np.conjugate(F_mat)
    A12 = 1j*(E_mat - np.conjugate(E_mat) + np.conjugate(F_mat) - F_mat)
    A21 = 1j*(np.conjugate(E_mat) - E_mat + np.conjugate(F_mat) - F_mat)
    A22 = E_mat + np.conjugate(E_mat) - F_mat - np.conjugate(F_mat)

    A = 0.5 * np.block([[A11, A12], [A21, A22]])

    cov = A @ A.T

    t0 = time.time()
    vacuum_density_matrix = twq.density_matrix(mu=np.array([0, 0, 0, 0]), cov=cov / 2, cutoff=M, hbar=1)
    vacuum_density_matrix = convert_density_matrix(vacuum_density_matrix)

    fu_temp = lambda omega: F(omega) * u(omega)
    gu_temp = lambda omega: G(2*Delta - omega) * u(2*Delta - omega).conjugate()

    zeta_u = np.sqrt(overlap(fu_temp, fu_temp, omegas))
    xi_u = np.sqrt(overlap(gu_temp, gu_temp, omegas))

    fu = lambda omega: fu_temp(omega) / zeta_u
    gu = lambda omega: gu_temp(omega) / xi_u

    b1 = qt.tensor(qt.destroy(M), qt.qeye(M))
    b2 = qt.tensor(qt.qeye(M), qt.destroy(M))

    plt.figure()
    plt.plot(omegas, fu(omegas) * np.conjugate(v1(omegas)))
    plt.plot(omegas, v1(omegas) * np.conjugate(gu(omegas)))
    plt.show()

    au = (zeta_u * overlap(fu, v1, omegas) * b1 - xi_u * overlap(v1, gu, omegas) * b1.dag()
          - zeta_u * np.sqrt(1 - overlap(fu, v1, omegas) * overlap(v1, fu, omegas)) * b2
          - xi_u * np.sqrt(1 - overlap(v1, gu, omegas) * overlap(gu, v1, omegas)) * b2.dag())

    density_matrix = au.dag() * vacuum_density_matrix * au
    t1 = time.time()
    print(f'Analytical two mode solution time={t1 - t0}')
    print('a1daga1', qt.expect(b1.dag() * b1, density_matrix))
    print('a2daga2', qt.expect(b2.dag() * b2, density_matrix))

    # Find the Wigner function numerically
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(qt.ptrace(density_matrix, 0), xvec, xvec)

    # normalize colors to the length of data
    nrm = mpl.colors.Normalize(w.min(), w.max())

    # Plot the Wigner function
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    # Find the Wigner function numerically
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(qt.ptrace(density_matrix, 1), xvec, xvec)

    # normalize colors to the length of data
    nrm = mpl.colors.Normalize(w.min(), w.max())

    # Plot the Wigner function
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    return density_matrix


def convert_density_matrix(density_matrix):
    new_density_matrix = np.zeros([M**2, M**2], dtype=np.complex128)
    for n1 in range(M):
        for m1 in range(M):
            for n2 in range(M):
                for m2 in range(M):
                    rhon1n2m1m2 = density_matrix[n1][m1][n2][m2]
                    new_density_matrix[n1 * M + n2, m1 * M + m2] = rhon1n2m1m2
    return qt.Qobj(new_density_matrix, dims=[[M, M], [M, M]])


def overlap(f, g, xs):
    return quad(lambda omega: np.conjugate(f(omega)) * g(omega), xs[0], xs[-1], complex_func=True)[0]


def g1_fock(omega1, omega2):
    term1 = n * F(omega1).conjugate() * u(omega1).conjugate() * F(omega2) * u(omega2)
    term2 = n * G(omega1) * u(2*Delta - omega1) * G(omega2).conjugate() * u(2*Delta - omega2).conjugate()
    return term1 + term2


def get_mode_coefs(u, f1_temp, g1_temp, f2_temp, g2_temp, xs) -> Tuple[List[float], List[float]]:
    """
    Computes the coefficients for the modes as in eq. 8 of the main paper for mode 1, and for mode 2 eq. 8 is
    also calculated, but then a further decomposition is performed to make it orthogonal to mode 1 as well.
    :return: Two lists of the coefficients of mode 1 and mode 2, and the angle for mixing with the final vacuum
    mode.
    """
    zeta1 = np.sqrt(overlap(f1_temp, f1_temp, xs))
    xi1 = np.sqrt(overlap(g1_temp, g1_temp, xs))

    zeta2 = np.sqrt(overlap(f2_temp, f2_temp, xs))
    xi2 = np.sqrt(overlap(g2_temp, g2_temp, xs))

    f1 = lambda omega: f1_temp(omega) / zeta1
    g1 = lambda omega: g1_temp(omega) / xi1

    f2 = lambda omega: f2_temp(omega) / zeta2
    g2 = lambda omega: g2_temp(omega) / xi2

    """ Getting all functions for v1 mode decomposition """

    uf1 = overlap(u, f1, xs)
    ug1 = overlap(u, g1, xs)

    h1 = lambda omega: (f1(omega) - u(omega) * uf1) / np.sqrt(1 - uf1 * uf1.conjugate())
    k1 = lambda omega: (g1(omega) - u(omega) * ug1) / np.sqrt(1 - ug1 * ug1.conjugate())

    k1h1 = overlap(k1, h1, xs)

    s1 = lambda omega: (h1(omega) - k1(omega) * k1h1) / np.sqrt(1 - k1h1 * k1h1.conjugate())

    """ Getting all functions for v2 mode decomposition """

    uf2 = overlap(u, f2, xs)
    ug2 = overlap(u, g2, xs)

    h2 = lambda omega: (f2(omega) - u(omega) * uf2) / np.sqrt(1 - uf2 * uf2.conjugate())
    k2 = lambda omega: (g2(omega) - u(omega) * ug2) / np.sqrt(1 - ug2 * ug2.conjugate())

    k2h2 = overlap(k2, h2, xs)
    s2 = lambda omega: (h2(omega) - k2(omega) * k2h2) / np.sqrt(1 - k2h2 * k2h2.conjugate())

    # Diagonalize with respect to 1-modes
    k1k2 = overlap(k1, k2, xs)
    k3 = lambda omega: (k2(omega) - k1(omega) * k1k2) / np.sqrt(1 - k1k2 * k1k2.conjugate())

    s1k3 = overlap(s1, k3, xs)
    k4 = lambda omega: (k3(omega) - s1(omega) * s1k3) / np.sqrt(1 - s1k3 * s1k3.conjugate())

    k1s2 = overlap(k1, s2, xs)
    s3 = lambda omega: (s2(omega) - k1(omega) * k1s2) / np.sqrt(1 - k1s2 * k1s2.conjugate())

    s1s3 = overlap(s1, s3, xs)
    s4 = lambda omega: (s3(omega) - s1(omega) * s1s3) / np.sqrt(1 - s1s3 * s1s3.conjugate())

    k4s4 = overlap(k4, s4, xs)

    """ Getting all coefficients """

    "First mode"

    A1 = zeta1 * uf1.conjugate()
    B1 = xi1 * ug1
    C1 = zeta1 * np.sqrt(1 - uf1.conjugate() * uf1) * k1h1.conjugate()
    D1 = xi1 * np.sqrt(1 - ug1.conjugate() * ug1)
    E1 = zeta1 * np.sqrt(1 - uf1.conjugate() * uf1) * np.sqrt(1 - k1h1.conjugate() * k1h1)

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

    return [A1, C1, E1, B1, D1], [A2, C2, E2, G2, B2, D2, F2, H2]


if __name__ == '__main__':
    main()
