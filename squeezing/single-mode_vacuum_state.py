import numpy as np
import qutip as qt
from scipy.integrate import quad
from util import math_functions as m
from util.bloch_messiah import SingleModeBlochMessiah
import matplotlib.pyplot as plt
import matplotlib as mpl
from thewalrus import quantum as twq

gamma = 1
xi = 0.1

tp = 4
tau = 1
N = 1
Delta = 0


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
    density_matrix1 = bloch_messiah_density_matrix()
    density_matrix2 = squeezed_vacuum_density_matrix()

    print(np.isclose(np.all(density_matrix1.full() - density_matrix2), 0))


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

    v_l = np.sqrt((zeta_u**2 + xi_u**2) + zeta_u * xi_u * fu_gu + zeta_u * xi_u * fu_gu.conjugate())
    return zeta_u, xi_u, f_u, g_u, v_l


def bloch_messiah_density_matrix():
    zeta_u, xi_u, f_u, g_u, v_l = get_parameters()
    v = lambda omega: (zeta_u * f_u(omega) + xi_u * g_u(omega)) / v_l

    # Use this mode to find the vacuum state using the Bloch-Messiah method
    f_temp = lambda omega: F(omega).conjugate() * v(omega)
    g_temp = lambda omega: gamma_N_minus(Delta - omega) * v(2*Delta - omega).conjugate()

    bm = SingleModeBlochMessiah(u, f_temp, g_temp, omegas, 12, qt.coherent(12, 0))
    rhov = bm.get_output_state()

    # Find the average number of photons in the mode
    a = qt.destroy(12)
    print(qt.expect(a.dag() * a, rhov))

    # Find the Wigner function numerically
    xvec = np.linspace(-5, 5, 200)
    w = qt.wigner(rhov, xvec, xvec)

    # normalize colors to the length of data
    nrm = mpl.colors.Normalize(w.min(), w.max())

    # Plot the Wigner function
    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(xvec, xvec, w, 100, cmap=mpl.cm.RdBu, norm=nrm)
    plt.colorbar(cs)
    plt.show()
    return rhov


def squeezed_vacuum_density_matrix():
    zeta_u, xi_u, f_u, g_u, v_l = get_parameters()
    v = lambda omega: (zeta_u * f_u(omega) + xi_u * g_u(omega)) / v_l

    # Use this mode to find the vacuum state using the Bloch-Messiah method
    f_temp = lambda omega: F(omega).conjugate() * v(omega)
    g_temp = lambda omega: gamma_N_minus(Delta - omega) * v(2 * Delta - omega).conjugate()

    zeta_v = np.sqrt(overlap(f_temp, f_temp))
    f_v = lambda x: f_temp(x) / zeta_v

    A1 = zeta_v * overlap(f_v, u)
    B1 = zeta_v * overlap(f_v, u) - 1/v_l
    C1 = zeta_v * np.sqrt(1 - overlap(f_v, u) * overlap(u, f_v))
    D1 = C1

    R11 = A1 + B1 + np.conjugate(A1) + np.conjugate(B1)
    R12 = C1 + D1 + np.conjugate(C1) + np.conjugate(D1)
    S11 = 1j * (A1 - B1 + np.conjugate(B1) - np.conjugate(A1))
    S12 = 1j * (C1 - D1 + np.conjugate(D1) - np.conjugate(C1))
    T11 = 1j * (np.conjugate(A1) + np.conjugate(B1) - A1 - B1)
    T12 = 1j * (np.conjugate(C1) + np.conjugate(D1) - C1 - D1)
    U11 = A1 - B1 + np.conjugate(A1) - np.conjugate(B1)
    U12 = C1 - D1 + np.conjugate(C1) - np.conjugate(D1)

    A = np.array([[R11, S11], [T11, U11]]) / 2
    B = np.array([[R12, S12], [T12, U12]]) / 2

    cov = A.T @ A + B @ B.T
    cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    alpha_vec = lambda x1, y1: np.array([[x1, y1]]).T

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

    density_matrix = twq.density_matrix(mu=np.array([0, 0]), cov=cov, cutoff=12)

    return density_matrix


if __name__ == '__main__':
    main()
