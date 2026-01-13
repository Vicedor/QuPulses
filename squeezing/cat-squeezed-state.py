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

alpha = 1 + 1j

tp = 4
tau = 1
N = 1
Delta = 0

M = 80


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
    density_matrix2 = vacuum_state_density_matrix()

    print(np.all(np.isclose(density_matrix1.full() - density_matrix2.full(), 0)))


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


def bloch_messiah_density_matrix():
    zeta_u, xi_u, f_u, g_u, v_l = get_parameters()
    v = lambda omega: (alpha * zeta_u * f_u(omega) + np.conjugate(alpha) * xi_u * g_u(omega)) / v_l

    # Use this mode to find the vacuum state using the Bloch-Messiah method
    f_temp = lambda omega: F(omega).conjugate() * v(omega)
    g_temp = lambda omega: gamma_N_minus(Delta - omega) * v(2*Delta - omega).conjugate()

    cat_state = (qt.displace(M, alpha) + 1j * qt.displace(M, -alpha)) * qt.basis(M, 0) / np.sqrt(2)

    bm = SingleModeBlochMessiah(u, f_temp, g_temp, omegas, M, cat_state)
    rhov = bm.get_output_state()
    print(rhov.full()[0, 0])

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


def vacuum_state_density_matrix():
    zeta_u, xi_u, f_u, g_u, v_l = get_parameters()
    v = lambda omega: (alpha * zeta_u * f_u(omega) + np.conjugate(alpha) * xi_u * g_u(omega)) / v_l

    # Use this mode to find the vacuum state using the Bloch-Messiah method
    f_temp = lambda omega: F(omega).conjugate() * v(omega)
    g_temp = lambda omega: gamma_N_minus(Delta - omega) * v(2 * Delta - omega).conjugate()

    zeta_v = np.sqrt(overlap(f_temp, f_temp))
    f_v = lambda x: f_temp(x) / zeta_v

    cov11 = 4 * zeta_v**2 + (alpha * np.conjugate(alpha)) / v_l**2 - 4 * zeta_v * np.real(alpha * overlap(f_v, u)) / v_l

    cov12 = - 2 *zeta_v * np.imag(overlap(f_v, u) * alpha) / v_l
    cov22 = alpha * np.conjugate(alpha) / v_l**2

    cov = np.array([[cov11, cov12],
                    [cov12, cov22]])

    density_matrix = qt.Qobj(twq.density_matrix(mu=np.array([0, 0]), cov=cov / 2, cutoff=M, normalize=True, hbar=1))

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


if __name__ == '__main__':
    main()
