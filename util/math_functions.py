"""
Implements several math functions often used
"""
import numpy as np
import qutip as qt
from scipy.special import erf
from scipy.integrate import quad, trapz
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


def two_mode_integral(tp: float, tau: float) -> Callable[[float], float]:
    f = gaussian_squared_integral(tp, tau)
    g = gaussian_sine_integral(tp, tau)
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
    f_with_fourier_factor = lambda w, tt: f(w) * np.exp(-1j * w * tt)
    f_real = lambda w, tt: np.real(f_with_fourier_factor(w, tt))  # real part
    f_imag = lambda w, tt: np.imag(f_with_fourier_factor(w, tt))  # imaginary part
    return quad(f_real, -np.inf, np.inf, args=(t,))[0] + 1j * quad(f_imag, -np.inf, np.inf, args=(t,))[0]
