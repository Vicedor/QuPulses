"""
Implements several math functions often used
"""
import numpy as np
import qutip as qt
from scipy.special import erf
from scipy.integrate import quad, trapz
from typing import Callable


def gaussian(tp: float, tau: float) -> Callable[[float], float]:
    """
    Returns a gaussian function with given tp and tau parameters
    :param tp: The offset in time of the gaussian
    :param tau: The width of the gaussian
    :return: A gaussian function with the given parameters
    """
    return lambda t: np.exp(-(t - tp) ** 2 / (2 * tau ** 2)) / (np.sqrt(tau) * np.pi ** 0.25)  # Square normalize


def gaussian_integral(tp: float, tau: float) -> Callable[[float], float]:
    """
    Returns a function which evaluates the integral of a gaussian function given the tp and tau parameters
    :param tp: The offset in time of the gaussian
    :param tau: The width of the gaussian
    :return: A function evaluating the integral of gaussian with the given parameters
    """
    return lambda t: 0.5 * (erf((t - tp) / tau) + erf(tp / tau))


def gaussian_sine(tp: float, tau: float) -> Callable[[float], float]:
    """
    A gaussian multiplied with a sine function to make it orthogonal to a regular gaussian
    :param tp: The offset in time of the gaussian and sine
    :param tau: The width of the gaussian
    :return: A gaussian * sine function handle
    """
    g = gaussian(tp, tau)
    return lambda t: g(t) * np.sin((t - tp)) * np.sqrt(2/(1 + np.exp(-tau**2)))  # last term is for normalization


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


def filtered_gaussian(tp: float, tau: float, gamma: float, w0: float, nT: int, times: np.ndarray):
    """
    Gets a function describing a filtered gaussian
    :param tp: The offset of gaussian pulse
    :param tau: The width of gaussian pulse
    :param gamma: The decay rate of cavity
    :param w0: The frequency of cavity
    :param nT: The size of the times array
    :param times: The array of times the function will be evaluated at
    :return: A cubic spline of the numerically calculated filtered gaussian function
    """
    v = _get_filtered_gaussian_as_list(tp, tau, gamma, w0, nT, times)

    # Return a cubic spline, so it is possible to evaluate at every given timestep
    v_real = qt.Cubic_Spline(times[0], times[-1], np.real(v))
    v_imag = qt.Cubic_Spline(times[0], times[-1], np.imag(v))
    return lambda t: v_real(t) + 1j * v_imag(t)


def _get_filtered_gaussian_as_list(tp, tau, gamma, w0, nT, times):
    """
    Calculates the filtered gaussian numerically as described short Kiilerich paper
    :param tp: The offset of gaussian pulse
    :param tau: The width of gaussian pulse
    :param gamma: The decay rate of cavity
    :param w0: The frequency of cavity
    :param nT: The size of the times array
    :param times: The array of times the function will be evaluated at
    :return: A list of the same length as the times list with the values of the filtered gaussian at these times
    """
    # Fourier transformed gaussian
    fourier_gaussian = lambda w, tt: np.sqrt(tau) * np.exp(-tau ** 2 * w ** 2 / 2 + 1j * tp * w) / (np.pi ** 0.25) \
                                     * np.exp(-1j * w * tt)
    # v(w) is giving as in eq. 7 in the letter through a Fourier transformation:
    v_w = lambda w, tt: (0.5 * gamma + 1j * (w - w0)) / (-0.5 * gamma + 1j * (w - w0)) * fourier_gaussian(w, tt)
    v_w_real = lambda w, tt: np.real(v_w(w, tt))  # real part
    v_w_imag = lambda w, tt: np.imag(v_w(w, tt))  # imaginary part

    # Calculate v for each timestep
    v = np.zeros(nT, dtype=np.complex_)
    for k in range(0, nT):
        v[k] = inverse_fourier_transform(v_w_real, v_w_imag, times[k])

    # Normalize v
    v = v / np.sqrt(trapz(v * np.conjugate(v), times))
    return v


def inverse_fourier_transform(f_real, f_imag, t):
    """
    Gives the inverse Fourier transform of f(w) to get f(t)
    :param f_real: the real part of fourier transformed f(w)
    :param f_imag: the imaginary part of fourier transformed f(w)
    :param t: The time at which to get v(t)
    :return: The inverse Fourier transformed v(t) at given time t
    """
    return quad(f_real, -np.inf, np.inf, args=(t,))[0] + 1j * quad(f_imag, -np.inf, np.inf, args=(t,))[0]


def filtered_gaussian_integral(tp, tau, gamma, w0, nT, times):
    """
    Calculates the integral of the norm-squared filtered gaussian
    :param tp: The offset of gaussian pulse
    :param tau: The width of gaussian pulse
    :param gamma: The decay rate of cavity
    :param w0: The frequency of cavity
    :param nT: The size of the times array
    :param times: The array of times the function will be evaluated at
    :return: A cubic spline of the norm-squared filtered gaussian
    """
    v_list = _get_filtered_gaussian_as_list(tp, tau, gamma, w0, nT, times)
    v2 = v_list * np.conjugate(v_list)
    v2_int = np.zeros(nT, dtype=np.complex_)
    for k in range(1, nT):
        intv2 = trapz(v2[0:k], times[0:k])
        v2_int[k] = intv2
    v_int_real = qt.Cubic_Spline(times[0], times[-1], np.real(v2_int))
    v_int_imag = qt.Cubic_Spline(times[0], times[-1], np.imag(v2_int))
    return lambda t: v_int_real(t) + 1j * v_int_imag(t)
