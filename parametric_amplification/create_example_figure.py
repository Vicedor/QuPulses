import numpy as np
import qutip as qt
from scipy.integrate import quad
from util import math_functions as m
import matplotlib.pyplot as plt
import matplotlib as mpl
from thewalrus import quantum as twq

from typing import Tuple, List


gamma = 1
xi = 0.1

alpha = 1 + 1j
n = 1

tp = 4
tau = 1
N = 1
Delta = 0

M = 50

# Array of frequencies
omegas = np.linspace(-4, 4, 1000)

# A gaussian input pulse in frequency domain (fourier transform of time domain)
u_tilde = lambda omega: np.sqrt(tau) / np.pi**(1/4) * np.exp(-tau**2 / 2 * omega**2 + 1j * tp * omega)
u = lambda omega: m.normalized_hermite_polynomial(tp, tau, 0)(omega) * u_tilde(omega)


# The functions needed for a TWPA transformation of input pulse
chi_X = lambda omega, xi: - ((gamma / 2 + xi)**2 + omega**2) / ((gamma/2 - 1j*omega)**2 - xi**2)
chi_P = lambda omega, xi: - ((gamma / 2 - xi)**2 + omega**2) / ((gamma/2 - 1j*omega)**2 - xi**2)

gamma_N_plus = lambda omega: (chi_X(omega, xi)**N + chi_P(omega, xi)**N) / 2
gamma_N_minus = lambda omega: (chi_X(omega, xi)**N - chi_P(omega, xi)**N) / 2

F = lambda omega: gamma_N_plus(omega - Delta)
G = lambda omega: gamma_N_minus(omega - Delta).conjugate()


def main():
    coherent_state_dm = squeezed_coherent_state_density_matrix()
    schrodinger_cat_dm = schrodinger_cat_density_matrix()
    single_photon_dm = single_photon_density_matrix()


def overlap(f, g, xs=omegas):
    """
    Calculates the inner product between functions f and g given by <f|g>
    :param f: Leftmost function in the inner product
    :param g: Rightmost function in the inner product
    :param xs: The axis which the functions are defined over
    :return: The value of the inner product
    """
    return quad(lambda omega: np.conjugate(f(omega)) * g(omega), xs[0], xs[-1], complex_func=True)[0]


def get_parameters():
    # In the following we use the "Note on parametric amplification vacuum states" to find the coherent state
    # output mode of the TWPA for alpha = 0
    f_u = lambda omega: F(omega) * u(omega)
    g_u = lambda omega: G(2*Delta - omega) * u(2*Delta - omega).conjugate()

    zeta_u = np.sqrt(overlap(f_u, f_u))
    xi_u = np.sqrt(overlap(g_u, g_u))

    f_u = lambda omega: F(omega) * u(omega) / zeta_u
    g_u = lambda omega: G(2*Delta - omega) * u(2*Delta - omega).conjugate() / xi_u

    fu_gu = overlap(f_u, g_u)

    v_l = np.sqrt(alpha * np.conjugate(alpha) * (zeta_u**2 + xi_u**2) + np.conjugate(alpha)**2 * zeta_u * xi_u * fu_gu + alpha ** 2 * zeta_u * xi_u * fu_gu.conjugate())
    return zeta_u, xi_u, f_u, g_u, v_l


def squeezed_coherent_state_density_matrix():
    zeta_u, xi_u, f_u, g_u, v_l = get_parameters()

    # Define v(omega) for a single-mode transformation (eq. 10 in the manuscript)
    v = lambda omega: (alpha * zeta_u * f_u(omega) + np.conjugate(alpha) * xi_u * g_u(omega)) / v_l

    # Define the fv and gv functions as integrals over F and G (delta function in omega', so integrals are easily found)
    f_temp = lambda omega: F(omega).conjugate() * v(omega)
    g_temp = lambda omega: gamma_N_minus(Delta - omega) * v(2 * Delta - omega).conjugate()

    # Normalize fv (gv is not used in the following, but could be normalized in the same way)
    zeta_v = np.sqrt(overlap(f_temp, f_temp))
    f_v = lambda x: f_temp(x) / zeta_v

    # Define the coherent addition to the covariant matrix eq. 37 in the manuscript
    cov11 = 16 * zeta_v**2 * np.real(alpha * overlap(f_v, u))**2 - 16 * zeta_v * alpha * np.conjugate(alpha) / v_l * np.real(alpha * overlap(f_v, u)) + 4 * (alpha * np.conjugate(alpha))**2 / v_l**2
    # Add the vacuum covariance matrix as in eq. 38 in the manuscript
    cov11 = cov11 + 4 * zeta_v**2 + (alpha * np.conjugate(alpha)) / v_l**2 - 4 * zeta_v * np.real(alpha * overlap(f_v, u)) / v_l

    cov12 = - 2 *zeta_v * np.imag(overlap(f_v, u) * alpha) / v_l
    cov22 = alpha * np.conjugate(alpha) / v_l**2

    # Define the displacement vector as in eq. 39 in the manuscript
    displace = np.array([np.sqrt(2)*(2 * zeta_v * np.real(alpha * overlap(f_v, u)) - alpha * np.conjugate(alpha) / v_l), 0])

    cov = np.array([[cov11, cov12],
                     [cov12, cov22]]) - 2 * np.outer(displace, displace)

    # In the following, we plot the Wigner function of the state with this covariance matrix
    cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    alpha_vec = lambda x1, y1: (np.array([[x1, y1]]) - displace).T

    xvec = np.linspace(-5, 5, 200)
    w = np.zeros((len(xvec), len(xvec)))

    for i, y1 in enumerate(xvec):
        for j, x1 in enumerate(xvec):
            val = (alpha_vec(x1, y1).T @ cov_inv @ alpha_vec(x1, y1))[0, 0]
            w[i, j] = np.exp(-val) / np.sqrt(det_cov) / np.pi

    nrm = mpl.colors.Normalize(w.min(), w.max())

    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    # We compute the density matrix of the state using multidimensional Hermite polynomials
    density_matrix = twq.density_matrix(mu=displace, cov=cov / 2, cutoff=M, normalize=True, hbar=1)

    return density_matrix


def schrodinger_cat_density_matrix():
    zeta_u, xi_u, f_u, g_u, v_l = get_parameters()

    # Define v(omega) for a single-mode transformation (eq. 10 in the manuscript)
    v = lambda omega: (alpha * zeta_u * f_u(omega) + np.conjugate(alpha) * xi_u * g_u(omega)) / v_l

    # Define the fv and gv functions as integrals over F and G (delta function in omega', so integrals are easily found)
    f_temp = lambda omega: F(omega).conjugate() * v(omega)
    g_temp = lambda omega: gamma_N_minus(Delta - omega) * v(2 * Delta - omega).conjugate()

    # Normalize fv (gv is not used in the following, but could be normalized in the same way)
    zeta_v = np.sqrt(overlap(f_temp, f_temp))
    f_v = lambda x: f_temp(x) / zeta_v

    # Define the covariance matrix as in eq. 38 in the manuscript
    cov11 = 4 * zeta_v**2 + (alpha * np.conjugate(alpha)) / v_l**2 - 4 * zeta_v * np.real(alpha * overlap(f_v, u)) / v_l
    cov12 = - 2 *zeta_v * np.imag(overlap(f_v, u) * alpha) / v_l
    cov22 = alpha * np.conjugate(alpha) / v_l**2

    cov = np.array([[cov11, cov12],
                    [cov12, cov22]])

    # Find the density matrix of the vacuum state using the multidimensional Hermite polynomials
    density_matrix = qt.Qobj(twq.density_matrix(mu=np.array([0, 0]), cov=cov / 2, cutoff=M, normalize=True, hbar=1))

    # Define the operator that creates a cat state from vacuum and apply to vacuum state eq. 41
    alpha_v = (2*zeta_u*np.real(alpha * overlap(v, f_u)) - alpha * np.conjugate(alpha) / v_l)
    cat_operator = (qt.displace(M, alpha_v) + 1j * qt.displace(M, -alpha_v)) / np.sqrt(2)

    density_matrix = cat_operator * density_matrix * cat_operator.dag()

    # Find the Wigner function numerically
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(density_matrix, xvec, xvec)

    # normalize colors to the length of data
    nrm = mpl.colors.Normalize(w.min(), w.max())

    # Plot the Wigner function
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()

    return density_matrix


def single_photon_density_matrix():
    v1, v2 = get_v1_and_v2()

    f1_temp = lambda omega: F(omega).conjugate() * v1(omega)
    g1_temp = lambda omega: G(omega) * v1(2*Delta - omega).conjugate()
    f2_temp = lambda omega: F(omega).conjugate() * v2(omega)
    g2_temp = lambda omega: G(omega) * v2(2*Delta - omega).conjugate()

    get_mode_coefs_alt(u, f1_temp, g1_temp, f2_temp, g2_temp, omegas)
    mode1_coefs, mode2_coefs = get_mode_coefs(u, f1_temp, g1_temp, f2_temp, g2_temp, omegas)

    A1, C1, E1, B1, D1 = mode1_coefs
    A2, C2, E2, B2, D2, F2 = mode2_coefs

    E_mat = np.array([[A1, C1, E1],
                      [A2, C2, E2]])
    F_mat = np.array([[B1, D1, 0],
                      [B2, D2, F2]])

    A11 = E_mat + np.conjugate(E_mat) + F_mat + np.conjugate(F_mat)
    A12 = 1j*(E_mat - np.conjugate(E_mat) + np.conjugate(F_mat) - F_mat)
    A21 = 1j*(np.conjugate(E_mat) - E_mat + np.conjugate(F_mat) - F_mat)
    A22 = E_mat + np.conjugate(E_mat) - F_mat - np.conjugate(F_mat)

    A = 0.5 * np.block([[A11, A12], [A21, A22]])

    cov = A @ A.T

    vacuum_density_matrix = twq.density_matrix(mu=np.array([0, 0, 0, 0]), cov=cov / 2, cutoff=M, hbar=1)

    def convert_density_matrix(density_matrix):
        new_density_matrix = np.zeros([M ** 2, M ** 2], dtype=np.complex128)
        for n1 in range(M):
            for m1 in range(M):
                for n2 in range(M):
                    for m2 in range(M):
                        rhon1n2m1m2 = density_matrix[n1][m1][n2][m2]
                        new_density_matrix[n1 * M + n2, m1 * M + m2] = rhon1n2m1m2
        return qt.Qobj(new_density_matrix, dims=[[M, M], [M, M]])

    vacuum_density_matrix = convert_density_matrix(vacuum_density_matrix)

    fu_temp = lambda omega: F(omega) * u(omega)
    gu_temp = lambda omega: G(2*Delta - omega) * u(2*Delta - omega).conjugate()

    zeta_u = np.sqrt(overlap(fu_temp, fu_temp, omegas))
    xi_u = np.sqrt(overlap(gu_temp, gu_temp, omegas))

    fu = lambda omega: fu_temp(omega) / zeta_u
    gu = lambda omega: gu_temp(omega) / xi_u

    gufu = overlap(gu, fu, omegas)
    fugu = np.conjugate(gufu)

    matrix = np.array([[n * zeta_u**2 + n * xi_u**2 * gufu * fugu, n*xi_u**2 * fugu * np.sqrt(1 - gufu*fugu)],
                       [n*xi_u**2 * gufu * np.sqrt(1 - gufu*fugu), n * xi_u**2 * (1 - fugu * gufu)]])

    #sqrt = np.sqrt(n**2 * (zeta_u**2 + xi_u**2)**2 - 4 * n**2 * zeta_u**2 * xi_u ** 2 * (1 - gufu * fugu))
    sqrt = np.sqrt(1 + ((zeta_u ** 2 + xi_u ** 2) ** 2 - 1) * fugu * gufu)
    coef1 = (n * zeta_u ** 2 + n * xi_u ** 2 * (2*gufu * fugu - 1) + sqrt) / (2 * n * xi_u**2 * gufu * np.sqrt(1 - gufu*fugu))
    coef2 = 1

    coef12 = (n * zeta_u ** 2 + n * xi_u ** 2 * (2*gufu * fugu - 1) - sqrt) / (2 * n * xi_u**2 * gufu * np.sqrt(1 - gufu*fugu))
    coef22 = 1

    norm2 = np.sqrt(sqrt / (2 * xi_u**4 * fugu * gufu * (1 - fugu * gufu)) * (sqrt - 1 - 2 * xi_u**2 * gufu*fugu))

    b1 = qt.tensor(qt.destroy(M), qt.qeye(M))
    b2 = qt.tensor(qt.qeye(M), qt.destroy(M))

    gufu = overlap(gu, fu, omegas)

    hu = lambda omega: (gu(omega) - gufu * fu(omega)) / np.sqrt(1 - gufu * np.conjugate(gufu))

    au = (zeta_u * overlap(fu, v1, omegas) * b1 - xi_u * overlap(v1, gu, omegas) * b1.dag()
          - zeta_u * np.sqrt(1 - overlap(fu, v1, omegas) * overlap(v1, fu, omegas)) * b2
          - xi_u * np.sqrt(1 - overlap(v1, gu, omegas) * overlap(gu, v1, omegas)) * b2.dag())

    density_matrix = au.dag() * vacuum_density_matrix * au

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


def get_v1_and_v2():
    fu_temp = lambda omega: F(omega) * u(omega)
    gu_temp = lambda omega: G(2*Delta - omega) * u(2*Delta - omega).conjugate()

    zeta_u = np.sqrt(overlap(fu_temp, fu_temp, omegas))
    xi_u = np.sqrt(overlap(gu_temp, gu_temp, omegas))

    fu = lambda omega: fu_temp(omega) / zeta_u
    gu = lambda omega: gu_temp(omega) / xi_u

    gufu = overlap(gu, fu, omegas)
    fugu = np.conjugate(gufu)

    hu = lambda omega: (gu(omega) - gufu * fu(omega)) / np.sqrt(1 - gufu * np.conjugate(gufu))

    val1 = (np.sqrt((n * (zeta_u**2 + xi_u**2))**2 - 4 * zeta_u**2 * xi_u**2 * (1 - gufu * np.conjugate(gufu)) * n**2) + n * (zeta_u**2 + xi_u**2)) / 2
    val2 = (-np.sqrt((n * (zeta_u ** 2 + xi_u ** 2))**2 - 4 * zeta_u ** 2 * xi_u ** 2 * (1 - gufu * np.conjugate(gufu)) * n ** 2) + n * (zeta_u ** 2 + xi_u ** 2)) / 2

    sqrt = np.sqrt(1 + ((zeta_u ** 2 + xi_u ** 2) ** 2 - 1) * fugu * gufu)
    norm1 = np.sqrt(sqrt / (2 * xi_u**4 * fugu * gufu * (1 - fugu * gufu)) * (sqrt + 1 + 2 * xi_u**2 * gufu*fugu))
    norm2 = np.sqrt(sqrt / (2 * xi_u**4 * fugu * gufu * (1 - fugu * gufu)) * (sqrt - 1 - 2 * xi_u**2 * gufu*fugu))

    coef1 = (1 + 2*xi_u**2 * fugu*gufu + sqrt) / (2 * xi_u**2 * gufu * np.sqrt(1 - fugu*gufu))
    coef2 = (1 + 2*xi_u**2 * fugu*gufu - sqrt) / (2 * xi_u**2 * gufu * np.sqrt(1 - fugu*gufu))

    v1 = lambda omega: coef1 / norm1 * fu(omega) + 1 / norm1 * hu(omega)
    v2 = lambda omega: coef2 / norm2 * fu(omega) + 1 / norm2 * hu(omega)

    norm_v1 = overlap(v1, v1, omegas)
    norm_v2 = overlap(v2, v2, omegas)

    v1_n = lambda omega: v1(omega) / np.sqrt(norm_v1)
    v2_n = lambda omega: v2(omega) / np.sqrt(norm_v2)

    return v1_n, v2_n


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

    # Other form of diagonalization

    A3 = zeta2 * overlap(f2, u, xs)
    C3 = zeta2 * overlap(f2, k1, xs)
    E3 = zeta2 * overlap(f2, s1, xs)

    B3 = xi2 * overlap(u, g2, xs)
    D3 = xi2 * overlap(k1, g2, xs)
    F3 = xi2 * overlap(s1, g2, xs)

    return [A1, C1, E1, B1, D1], [A3, C3, E3, B3, D3, F3] #[A2, C2, E2, G2, B2, D2, F2, H2]


def get_mode_coefs_alt(u, f1_temp, g1_temp, f2_temp, g2_temp, xs) -> Tuple[List[float], List[float]]:
    zeta1 = np.sqrt(overlap(f1_temp, f1_temp, xs))
    xi1 = np.sqrt(overlap(g1_temp, g1_temp, xs))

    zeta2 = np.sqrt(overlap(f2_temp, f2_temp, xs))
    xi2 = np.sqrt(overlap(g2_temp, g2_temp, xs))

    f1 = lambda omega: f1_temp(omega) / zeta1
    g1 = lambda omega: g1_temp(omega) / xi1

    f2 = lambda omega: f2_temp(omega) / zeta2
    g2 = lambda omega: g2_temp(omega) / xi2

    uf2 = overlap(u, f2, xs)
    h = lambda omega: (f2(omega) - uf2 * u(omega)) / np.sqrt(1 - uf2 * np.conjugate(uf2))

    ug2 = overlap(u, g2, xs)
    hg2 = overlap(h, g2, xs)

    k = lambda omega: (g2(omega) - ug2 * u(omega) - hg2 * h(omega)) / np.sqrt(1 - ug2 * np.conjugate(ug2) - hg2 * np.conjugate(hg2))


if __name__ == '__main__':
    main()
