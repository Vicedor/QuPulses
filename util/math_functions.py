"""
Implements several math functions often used
"""
import math
import numpy as np
import qutip as qt
from scipy.special import erf
from scipy.integrate import quad, trapz, complex_ode
from typing import Callable, List


def exponential(g: float) -> Callable[[float], float]:
    return lambda t: np.exp(-g**2*t/2)


def exponential_integral(g: float) -> Callable[[float], float]:
    return lambda t: (1 - np.exp(-g**2*t))/g**2


def gaussian(tp: float, tau: float) -> Callable[[float], float]:
    """
    Returns a gaussian function with given tp and tau parameters
    :param tp: The offset in time of the gaussian
    :param tau: The width of the gaussian
    :return: A gaussian function with the given parameters
    """
    return lambda t: np.exp(-(t - tp) ** 2 / (2 * tau ** 2)) / (np.sqrt(tau) * np.pi ** 0.25)  # Square normalize


def freq_mod_gaussian(tp: float, tau: float, w: float) -> Callable[[float], float]:
    g = gaussian(tp, tau)
    return lambda t: np.exp(1j*w*t) * g(t)


def gaussian_integral(tp: float, tau: float) -> Callable[[float], float]:
    """
    Returns a function which evaluates the integral of a gaussian function given the tp and tau parameters
    :param tp: The offset in time of the gaussian
    :param tau: The width of the gaussian
    :return: A function evaluating the integral of gaussian with the given parameters
    """
    a = np.pi ** 0.25 * np.sqrt(tau) / np.sqrt(2)
    return lambda t: a * (erf((t - tp)/(np.sqrt(2) * tau)) + erf(tp/(np.sqrt(2) * tau)))


def gaussian_squared_integral(tp: float, tau: float) -> Callable[[float], float]:
    """
    Returns a function which evaluates the integral of the square of a gaussian function given the tp and tau parameters
    :param tp: The offset in time of the gaussian
    :param tau: The width of the gaussian
    :return: A function evaluating the integral of gaussian with the given parameters
    """
    return lambda t: 0.5 * (erf((t - tp) / tau) + erf(tp / tau))


def hermite_gaussian_integral(tp: float, tau: float, order: int) -> Callable[[float], float]:
    """
    Returns a function which evaluates the integral of the square of the hermite gaussian for the given order and the
    given tp and tau parameters, from 0 up to the given times
    :param tp: The offset in time of the hermite gaussian
    :param tau: The width of the gaussian
    :param order: The order of hermite polynomial
    :return: A function evaluating the integral of the squared hermite polynomial of the given order from 0 to t
    """
    if order == 0:
        return gaussian_squared_integral(tp, tau)
    if order == 1:
        return lambda t: erf((t - tp)/tau) / 2 - (t - tp)/tau * np.exp(-((t - tp)/tau)**2) / np.sqrt(np.pi) + 0.5
    if order == 2:
        return lambda t: erf((t - tp)/tau) / 2 - np.exp(-((t - tp)/tau)**2) * (((t - tp)/tau)**3
                                                                               + (t - tp)/(2*tau)) / np.sqrt(np.pi)\
                         + 0.5


def two_mode_integral(tp: float, tau: float) -> Callable[[float], float]:
    f = gaussian_squared_integral(tp, tau)
    g = hermite_gaussian_integral(tp, tau, 1)
    return lambda t: (f(t) + g(t)) / 2


def gaussian_sine(tp: float, tau: float) -> Callable[[float], float]:
    """
    A gaussian multiplied with a sine function to make it orthogonal to a regular gaussian
    :param tp: The offset in time of the gaussian and sine
    :param tau: The width of the gaussian
    :return: A gaussian * sine function handle
    """
    g = gaussian(tp, tau)
    return lambda t: g(t) * np.sin((t - tp)) * np.sqrt(2/(1 - np.exp(-tau**2)))  # last term is for normalization


def gaussian_sine_integral(tp: float, tau: float) -> Callable[[float], float]:
    """
    Returns a function that evaluates the integral of a gaussian times a sine function from 0 to t
    :param tp: The offset in time of the gaussian and sine
    :param tau: The width of the gaussian
    :return: A function that evaluate the integral of gaussian * sine from 0 to t
    """
    tau_sq = tau ** 2

    def temp(t: float):
        a = erf((t + 1j*tau_sq - tp)/tau)
        b = erf((t - 1j*tau_sq - tp)/tau)
        c = 2*np.exp(tau_sq)*erf((t - tp)/tau)
        d = erf((1j*tau_sq + tp)/tau)
        e = erf((1j*tau_sq - tp)/tau)
        f = 2 * erf(tp/tau) * np.exp(tau_sq)
        g = 4*np.exp(tau_sq) - 4
        return - (a + b - c + d - e - f) / g
    return temp


def normalized_hermite_polynomial(tp: float, tau: float, order: int) -> Callable[[float], float]:
    """
    Returns a function evaluating a normalized Hermite polynomial (of the physicist's kind from wikipedia
    https://www.wikiwand.com/en/Hermite_polynomials). The offset determines the midpoint of polynomial, while the
    order gives the order of the polynomial
    The normalization factor is determined such that the integral from -inf to inf over H_i(x) * g(x) = 1 where
    g(x) is a normalized gaussian distribution
    :param tp: The offset in x for the midpoint of the polynomial (default is 0)
    :param tau: The width of the gaussian it is multiplied with (needed for normalization)
    :param order: The order of the polynomial H_i(x) where i is the order (minimum is i = 0, maximum is i = 10)
    :return: A function handle for the given order of Hermite polynomial with the given offset
    """
    norm_factor = np.sqrt(2**order * math.factorial(order))
    if order == 0:
        return lambda x: 1
    if order == 1:
        return lambda x: (2 * ((x - tp)/tau)) / norm_factor
    if order == 2:
        return lambda x: (4 * ((x - tp)/tau) ** 2 - 2) / norm_factor
    if order == 3:
        return lambda x: (8 * ((x - tp)/tau) ** 3 - 12 * ((x - tp)/tau)) / norm_factor
    if order == 4:
        return lambda x: (16 * ((x - tp)/tau) ** 4 - 48 * ((x - tp)/tau) ** 2 + 12) / norm_factor
    if order == 5:
        return lambda x: (32 * ((x - tp)/tau) ** 5 - 160 * ((x - tp)/tau) ** 3 + 120 * ((x - tp)/tau)) / norm_factor
    if order == 6:
        return lambda x: (64 * ((x - tp)/tau) ** 6 - 480 * ((x - tp)/tau) ** 4 + 720 * ((x - tp)/tau) ** 2
                          - 120) / norm_factor
    if order == 7:
        return lambda x: (128 * ((x - tp)/tau) ** 7 - 1344 * ((x - tp)/tau) ** 5 + 3360 * ((x - tp)/tau) ** 3
                          - 1680 * ((x - tp)/tau)) / norm_factor
    if order == 8:
        return lambda x: (256 * ((x - tp)/tau) ** 8 - 3584 * ((x - tp)/tau) ** 6 + 13440 * ((x - tp)/tau) ** 4
                          - 13440 * ((x - tp)/tau) ** 2 + 1680)/norm_factor
    if order == 9:
        return lambda x: (512 * ((x - tp)/tau) ** 9 - 9216 * ((x - tp)/tau) ** 7 + 48384 * ((x - tp)/tau) ** 5
                          - 80640 * ((x - tp)/tau) ** 3 + 30240 * ((x - tp)/tau)) / norm_factor
    if order == 10:
        return lambda x: (1024 * ((x - tp)/tau) ** 10 - 23040 * ((x - tp)/tau) ** 8 + 161280 * ((x - tp)/tau) ** 6
                          - 403200 * ((x - tp)/tau) ** 4 + 302400 * ((x - tp)/tau) ** 2 - 30240) / norm_factor
    else:
        raise ValueError(f"Order only defined up to 10, and not for order={order}!")


def numerical_reflected_mode(um: Callable[[float], float], un: Callable[[float], float]) -> Callable[[float], float]:
    """
    Calculates numerically what the incoming mode should be, to turn into the mode um(t) after reflection on an input
    cavity which itself emits a mode given by un(t)
    :param um: The desired mode-shape after reflection
    :param un: The mode shape emitted by the input cavity reflected upon
    :return: The mode which will turn into un(t) after reflection
    """
    int1 = lambda t: quad(lambda t_prime: np.conjugate(un(t_prime)) * um(t_prime), 0, t)[0]
    int2 = lambda t: quad(lambda t_prime: np.conjugate(un(t_prime)) * un(t_prime), 0, t)[0]
    return lambda t: um(t) + un(t) * int1(t) / (1e-6 + 1 - int2(t))


def fourier_gaussian(tp: float, tau: float):
    """
    Gets the fourier transform of a gaussian as a function with parameter omega.
    :param tp: The time the pulse peaks
    :param tau: The width of the gaussian
    :return: A function which evaluates the fourier transform of a gaussian at a given frequency
    """
    return lambda w: np.sqrt(tau) * np.exp(-tau ** 2 * w ** 2 / 2 + 1j * tp * w) / (np.pi ** 0.25)


def theta(t, tp, tau):
    """
    Analytical derivation of the antiderivative of -1/2 g_u(t) * g_v(t) for u(t) = v(t) and u(t) is gaussian
    :param t: The time
    :param tp: The pulse peak
    :param tau: The pulse width
    :return: The value of the analytical antiderivative
    """
    return - np.arcsin(np.sqrt((erf((t - tp) / tau) + erf(tp / tau)) / 2))


def cot(t):
    """
    The cotangent of the angle t: cot(t) = cos(t)/sin(t). If sin(t) = 0 it returns cot(t) = 0
    :param t: the angle
    :return: the cotangent of the angle
    """
    if isinstance(t, float):
        if np.sin(t) == 0:
            return 0
    return np.cos(t) / np.sin(t)


def filtered_gaussian(tp: float, tau: float, gamma: float, w0: float, times: np.ndarray):
    """
    Gets a function describing a filtered gaussian
    :param tp: The offset of gaussian pulse
    :param tau: The width of gaussian pulse
    :param gamma: The decay rate of cavity
    :param w0: The frequency of cavity
    :param times: The array of times the function will be evaluated at
    :return: A cubic spline of the numerically calculated filtered gaussian function
    """
    v = get_filtered_gaussian_as_list(tp, tau, gamma, w0, times)

    # Return a cubic spline, so it is possible to evaluate at every given timestep
    v_t = qt.Cubic_Spline(times[0], times[-1], v)
    return v_t


def filtered_gaussian_integral(tp, tau, gamma, w0, times):
    """
    Calculates the integral of the norm-squared filtered gaussian
    :param tp: The offset of gaussian pulse
    :param tau: The width of gaussian pulse
    :param gamma: The decay rate of cavity
    :param w0: The frequency of cavity
    :param times: The array of times the function will be evaluated at
    :return: A cubic spline of the norm-squared filtered gaussian
    """
    v_list = get_filtered_gaussian_as_list(tp, tau, gamma, w0, times)
    v2 = v_list * np.conjugate(v_list)
    nT = len(times)
    v2_int = np.zeros(nT, dtype=np.complex_)
    for k in range(1, nT):
        intv2 = trapz(v2[0:k], times[0:k])
        v2_int[k] = intv2
    v_int = qt.Cubic_Spline(times[0], times[-1], v2_int)
    return v_int


def get_filtered_gaussian_as_list(tp: float, tau: float, gamma: float, w0: float, times: np.ndarray):
    """
    Gets the filtered gaussian as a list of function values evaluated at the given times
    :param tp: The offset of gaussian pulse
    :param tau: The width of gaussian pulse
    :param gamma: The decay rate of cavity
    :param w0: The frequency of cavity
    :param times: The array of times the function will be evaluated at
    :return: A list of the filtered gaussian evaluated at the times given in the times array
    """
    # Fourier transformed gaussian
    fourier_gaussian_w = fourier_gaussian(tp, tau)
    # v(w) is giving as in eq. 7 in the letter through a Fourier transformation:
    dispersion_factor = lambda w: (0.5 * gamma + 1j * (w - w0)) / (-0.5 * gamma + 1j * (w - w0))
    v_w = lambda w: dispersion_factor(w) * fourier_gaussian_w(w)

    return _get_inverse_fourier_transform_as_list(v_w, times)


def n_filtered_gaussian(tp: float, tau: float, gammas: List[float], w0s: List[float], times: np.ndarray):
    """
    Gets the filtered gaussian temporal mode after passing through n cavities
    :param tp: The offset of gaussian pulse
    :param tau: The width of gaussian pulse
    :param gammas: List of the decay rates of cavities
    :param w0s: List of the frequencies of cavities
    :param times: The array of times the function will be evaluated at
    :return: A cubic spline of the norm-squared n-filtered gaussian
    """
    v = get_n_filtered_gaussian_as_list(tp, tau, gammas, w0s, times)

    # Return a cubic spline, so it is possible to evaluate at every given timestep
    v_t = qt.Cubic_Spline(times[0], times[-1], v)
    return v_t


def n_filtered_gaussian_integral(tp: float, tau: float, gammas: List[float], w0s: List[float], times: np.ndarray):
    """
    Calculates the integral of the norm-squared n-filtered gaussian
    :param tp: The offset of gaussian pulse
    :param tau: The width of gaussian pulse
    :param gammas: List of the decay rates of cavities
    :param w0s: List of the frequencies of cavities
    :param times: The array of times the function will be evaluated at
    :return: A cubic spline of the norm-squared n-filtered gaussian
    """
    v_list = get_n_filtered_gaussian_as_list(tp, tau, gammas, w0s, times)
    v2 = v_list * np.conjugate(v_list)
    nT = len(times)
    v2_int = np.zeros(nT, dtype=np.complex_)
    for k in range(1, nT):
        intv2 = trapz(v2[0:k], times[0:k])
        v2_int[k] = intv2
    v_int = qt.Cubic_Spline(times[0], times[-1], v2_int)
    return v_int


def get_n_filtered_gaussian_as_list(tp: float, tau: float, gammas: List[float], w0s: List[float], times: np.ndarray):
    """
    Gets the n-filtered gaussian as a list of function values evaluated at the given times
    :param tp: The offset of gaussian pulse
    :param tau: The width of gaussian pulse
    :param gammas: List of the decay rates of cavities
    :param w0s: List of the frequencies of cavities
    :param times: The array of times the function will be evaluated at
    :return: A list of the n-filtered gaussian evaluated at the times given in the times array
    """
    # Fourier transformed gaussian
    fourier_gaussian_w = fourier_gaussian(tp, tau)
    # v(w) is giving as in eq. 7 in the letter through a Fourier transformation:
    dispersion_factor = lambda w, gamma, w0: (0.5 * gamma + 1j * (w - w0)) / (-0.5 * gamma + 1j * (w - w0))

    def v_w(w):
        out = fourier_gaussian_w(w)
        for i, gamma in enumerate(gammas):
            w0 = w0s[i]
            out *= dispersion_factor(w, gamma, w0)

        return out

    return _get_inverse_fourier_transform_as_list(v_w, times)


def _get_inverse_fourier_transform_as_list(f_w: Callable[[float], float], times):
    """
    Calculates the inverse fourier transform numerically and returns a list of the function evaluated at the given times
    :param f_w: The fourier transformed function to be taken the inverse fourier transform of
    :param times: The array of times the function will be evaluated at
    :return: A list of the same length as the times list with the values of the inverse fourer transform at these times
    """
    # Calculate f for each timestep
    nT = len(times)
    f_t = np.zeros(nT, dtype=np.complex_)
    for k in range(0, nT):
        f_t[k] = inverse_fourier_transform(f_w, times[k])

    # Normalize f_t
    f_t = f_t / np.sqrt(trapz(f_t * np.conjugate(f_t), times))
    return f_t


def inverse_fourier_transform(f: Callable[[float], float], t: float):
    """
    Gives the inverse Fourier transform of f(w) to get f(t)
    :param f: the function to perform inverse fourier transform on f(w)
    :param t: The time at which to get v(t)
    :return: The inverse Fourier transformed v(t) at given time t
    """
    f_with_fourier_factor = lambda w, tt: f(w) * np.exp(-1j * w * tt) / np.sqrt(2*np.pi)
    f_real = lambda w, tt: np.real(f_with_fourier_factor(w, tt))  # real part
    f_imag = lambda w, tt: np.imag(f_with_fourier_factor(w, tt))  # imaginary part
    return quad(f_real, -np.inf, np.inf, args=(t,))[0] + 1j * quad(f_imag, -np.inf, np.inf, args=(t,))[0]


"""Solve Interaction picture numerically through numerical solution to differential equation"""


def matrix2vec(A):
    """
    Converts a nxn matrix to a n^2x1 vector
    :param A: The nxn matrix
    :return: The n^2x1 vector
    """
    return A.flatten()


def vec2matrix(v):
    """
    Converts a n^2x1 vector to a nxn matrix
    :param v: The n^2x1 vector
    :return: The nxn matrix
    """
    n = int(np.sqrt(len(v)))
    return v.reshape(n, n)


def solve_numerical_interaction_picture(F: Callable[[float], np.ndarray], M0: np.ndarray,
                                        times: np.ndarray) -> List[List[qt.Cubic_Spline]]:
    """
    Solves a matrix differential equation for a set of operators, such as achieved from solving an interaction picture.
    The equation is of the form vec(a(t)) = M(t) vec(a(0)), where d/dt M(t) = F(t) * M(t) and M(t) is an n x n matrix.
    :param F: The matrix which couples the entries of M(t) to each other in the equation d/dt M(t) = F(t) * M(t)
    :param M0: The initial condition for the M(t = 0) matrix.
    :param times: The times at which the solution to the differential equation shall be found
    :return: A list of splines, one for each entrance in the matrix M(t)
    """
    nT = len(times)
    T = times[-1]

    n = M0.shape[0]

    def deriv(t, U):
        U = vec2matrix(U)
        dUdt = F(t) @ U
        return matrix2vec(dUdt)

    ode = complex_ode(deriv)
    ode.set_initial_value(matrix2vec(M0))

    Ulist = np.zeros((nT, n**2), dtype=np.complex_)

    dt = T / nT
    for i in range(nT):
        if ode.successful():
            Ulist[i] = ode.integrate(ode.t + dt)

    Usplines: List[List] = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            Usplines[i][j] = qt.Cubic_Spline(0, T, Ulist[:, i*n + j])
    return Usplines


def get_time_dependent_modes(U: List[List[qt.Cubic_Spline]], a0: List[qt.Qobj]) -> List[qt.QobjEvo]:
    """
    Converts a matrix U of time-dependent functions and a set of qutip operators a0 to a set of time dependent
    operators a(t), as defined by the matrix product a(t) = U(t) * a0. This function is used in conjunction with the
    output from solve_numerical_interaction_picture to get the actual time-dependent operators
    :param U: The time-dependent matrix where the entries are time-dependent functions
    :param a0: The initial operators at t = 0
    :return: The operators as a function of time a(t)
    """
    a_t: List[qt.QobjEvo] = []
    n = len(a0)
    for i in range(n):
        a_t.append(qt.QobjEvo([[a0[j], U[i][j]] for j in range(n)]))
    return a_t


"""Reflected modes"""


def hermite_gaussian_reflected_on_gaussian(tp: float, tau: float, order: int) -> Callable[[float], float]:
    if order == 1:
        un = gaussian(tp, tau)
        h = normalized_hermite_polynomial(tp, tau, order)
        um = lambda t: un(t) * h(t)
        g = gaussian_squared_integral(tp, tau)
        return lambda t: um(t) - un(t) * np.sqrt(2/np.pi) * np.exp(-((t - tp)/tau)**2) / 2 / (10e-6 + 1 - g(t))
