from typing import Union, List, Callable

import numpy as np
import qutip as qt
from scipy.special import erf
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid, quad
import matplotlib.pyplot as plt

epsilon = 1e-10  # Kill rate to avoid numerical instability

N = 3
d = 2
M = 3

N2 = 21
M2 = 3
offset = 14

tau = 1
tp = 4
w0 = 0
gamma = 1

times = np.linspace(0, 12, 1000)

gauss = 'gauss'
filtered_gauss = 'filtered_gauss'


def main():
    input_output = Input_Output()
    result: qt.solver.Result = integrate_master_equation(input_output.liouvillian, input_output.psi0,
                                                         c_ops=[], e_ops=input_output.get_expectation_observables(),
                                                         times=times)

    plt.figure()
    plt.plot(times, result.expect[0])
    plt.plot(times, result.expect[1])
    plt.plot(times, result.expect[2])
    plt.show()

    input_output_int = Interaction_Picture_Input_Output()
    result: qt.solver.Result = integrate_master_equation(input_output_int.liouvillian, input_output_int.psi0,
                                                         c_ops=[], e_ops=input_output_int.get_expectation_observables(),
                                                         times=times)

    plt.figure()
    plt.plot(times, result.expect[0])
    plt.plot(times, result.expect[1])
    plt.plot(times, result.expect[2])
    plt.show()


def gaussian(tp: float, tau: float) -> Callable[[float], float]:
    """
    Returns a gaussian function with given tp and tau parameters
    :param tp: The offset in time of the gaussian
    :param tau: The width of the gaussian
    :return: A gaussian function with the given parameters
    """
    return lambda t: np.exp(-(t - tp) ** 2 / (2 * tau ** 2)) / (np.sqrt(tau) * np.pi ** 0.25)  # Square normalize


def gaussian_squared_integral(tp: float, tau: float) -> Callable[[float], float]:
    """
    Returns a function which evaluates the integral of the square of a gaussian function given the tp and tau parameters
    :param tp: The offset in time of the gaussian
    :param tau: The width of the gaussian
    :return: A function evaluating the integral of gaussian with the given parameters
    """
    return lambda t: 0.5 * (erf((t - tp) / tau) + erf(tp / tau))


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
    v_t = CubicSpline(times, v)
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
    v2_int = np.zeros(nT, dtype=np.complex128)
    for k in range(1, nT):
        intv2 = trapezoid(v2[0:k], times[0:k])
        v2_int[k] = intv2
    v_int = CubicSpline(times, v2_int)
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

    return get_inverse_fourier_transform_as_list(v_w, times)


def fourier_gaussian(tp: float, tau: float):
    """
    Gets the fourier transform of a gaussian as a function with parameter omega.
    :param tp: The time the pulse peaks
    :param tau: The width of the gaussian
    :return: A function which evaluates the fourier transform of a gaussian at a given frequency
    """
    return lambda w: np.sqrt(tau) * np.exp(-tau ** 2 * w ** 2 / 2 + 1j * tp * w) / (np.pi ** 0.25)


def get_inverse_fourier_transform_as_list(f_w: Callable[[float], float], times):
    """
    Calculates the inverse fourier transform numerically and returns a list of the function evaluated at the given times
    :param f_w: The fourier transformed function to be taken the inverse fourier transform of
    :param times: The array of times the function will be evaluated at
    :return: A list of the same length as the times list with the values of the inverse fourier transform at these times
    """
    # Calculate f for each timestep
    nT = len(times)
    f_t = np.zeros(nT, dtype=np.complex128)
    for k in range(0, nT):
        f_t[k] = inverse_fourier_transform(f_w, times[k])

    # Normalize f_t
    f_t = f_t / np.sqrt(trapezoid(f_t * np.conjugate(f_t), times))
    return f_t


def inverse_fourier_transform(f: Callable[[float], float], t: float) -> complex:
    """
    Gives the inverse Fourier transform of f(w) to get f(t)
    :param f: the function f(w) to perform inverse fourier transform
    :param t: The time at which to get f(t)
    :return: The inverse Fourier transformed f(t) at given time t
    """
    f_with_fourier_factor = lambda w, tt: f(w) * np.exp(-1j * w * tt) / np.sqrt(2*np.pi)
    f_real = lambda w, tt: np.real(f_with_fourier_factor(w, tt))  # real part
    f_imag = lambda w, tt: np.imag(f_with_fourier_factor(w, tt))  # imaginary part
    return quad(f_real, -np.inf, np.inf, args=(t,))[0] + 1j * quad(f_imag, -np.inf, np.inf, args=(t,))[0]


def theta(t):
    return np.arcsin(np.sqrt(gaussian_squared_integral(tp, tau)(t)))


def cot(x):
    if x != 0:
        return np.cos(x) / np.sin(x)
    else:
        return 0


def csc(x):
    if x != 0:
        return 1 / np.sin(x)
    else:
        return 0


def integrate_master_equation(f: Union[qt.Qobj, qt.QobjEvo, Callable[[float, any], qt.Qobj]], psi: qt.Qobj,
                              c_ops: List[qt.Qobj], e_ops: List[qt.Qobj], times: np.ndarray,
                              options: qt.Options = qt.Options(nsteps=1000000000, store_states=1, atol=1e-8, rtol=1e-6),
                              verbose=True) -> qt.solver.Result:
    """
    Integrates the master equation for the system specifications specified in the setup.py file
    :param f: Something to evaluate a quantum system, either Hamiltonian or Liouvillian
    :param psi: The initial state as a ket
    :param c_ops: The collapse operators for the system (not necessary if defined in Liouvillian)
    :param e_ops: The observables to be tracked during the time-evolution
    :param times: An array of the times to evaluate the observables at
    :param options: The options for the integrator, as a qutip Options object
    :param verbose: Whether to display a progress bar or not. Default: True
    :return: The expectation values of the number operators for the ingoing pulse, outgoing pulse and system excitations
             in that order
    """
    output = qt.mesolve(f, psi, tlist=times, c_ops=c_ops, e_ops=e_ops, progress_bar=verbose, options=options)
    return output


class Pulse:
    def __init__(self, shape, in_going: bool, args):
        self.shape = shape
        self.in_going: bool = in_going
        if self.shape == gauss:
            u = gaussian(*args)
            g = gaussian_squared_integral(*args)
        elif self.shape == filtered_gauss:
            u = filtered_gaussian(*args)
            g = filtered_gaussian_integral(*args)
        else:
            raise NotImplemented('This pulse shape is not implemented (yet)')
        self._u, self._g = u, g

    def u(self, t: float) -> float:
        """
        Evaluates the pulse shape u(t) depending on the pulse shape specified
        :param t: The time to evaluate the pulse shape at
        :return: The normalized pulse shape value at the specified time
        """
        return self._u(t)

    def g(self, t: float) -> float:
        """
        Evaluates the g(t) function (eq. 2 in the InteractionPicturePulses paper) given the specified pulse shape
        :param t: The time at which the function is evaluated
        :return: The value of g_u(t) at the specified time
        """
        real = False
        temp = np.conjugate(self.u(t))
        if np.imag(temp) == 0:
            real = True
        # Divide by the integral of u(t) only if u(t) is not very small (to avoid numerical instability)
        if abs(temp) >= epsilon:
            if self.in_going:
                temp /= np.sqrt(epsilon + 1 - self._g(t))
            else:
                temp = - temp / np.sqrt(epsilon + self._g(t))
        # If t=0 the out_going g(t) function is infinite, so set it to 0 to avoid numerical instability
        if not self.in_going and abs(temp) >= 1/epsilon:
            temp = 0
        if real:
            temp = np.real(temp)
        return temp


class Input_Output:
    def __init__(self):
        self.psi0 = qt.tensor(qt.basis(N, 1), qt.basis(d, 0), qt.basis(M, 0))

        au = qt.destroy(N)
        c = qt.destroy(d)
        av = qt.destroy(M)
        Iu = qt.qeye(N)
        Id = qt.qeye(d)
        Iv = qt.qeye(M)
        self.au_tot = qt.tensor(au, Id, Iv)
        self.c_tot = qt.tensor(Iu, c, Iv)
        self.av_tot = qt.tensor(Iu, Id, av)
        self.I = qt.tensor(Iu, Id, Iv)

        self.u_pulse = Pulse(shape=gauss, in_going=True, args=[tp, tau])
        self.v_pulse = Pulse(shape=filtered_gauss, in_going=False, args=[tp, tau, gamma, w0, times])
        self.H = qt.QobjEvo([[0.5j * np.sqrt(gamma) * (self.au_tot.dag() * self.c_tot - self.c_tot.dag() * self.au_tot),
                              lambda t, args: self.u_pulse.g(t)],
                             [0.5j * np.sqrt(gamma) * (self.c_tot.dag() * self.av_tot - self.av_tot.dag() * self.c_tot),
                              lambda t, args: self.v_pulse.g(t)],
                             [0.5j * (self.au_tot.dag() * self.av_tot - self.av_tot.dag() * self.au_tot),
                              lambda t, args: self.u_pulse.g(t) * self.v_pulse.g(t)]])
        self.L = [qt.QobjEvo([[self.au_tot, lambda t, args: self.u_pulse.g(t)]]) + np.sqrt(gamma) * self.c_tot
                  + qt.QobjEvo([[self.av_tot, lambda t, args: self.v_pulse.g(t)]])]

    def get_expectation_observables(self) -> Union[List[qt.Qobj], Callable]:
        return [self.au_tot.dag() * self.au_tot, self.c_tot.dag() * self.c_tot, self.av_tot.dag() * self.av_tot]

    def liouvillian(self, t: float, args) -> qt.Qobj:
        """
        Gets the liouvillian from the reduced Hamiltonian and collapse operators. This should only be used on the final
        total component for the whole SLH-network
        :return: The liouvillian ready to use for a master equation solver
        """
        Ls = [row(t) if isinstance(row, qt.QobjEvo) else row for row in self.L]
        if isinstance(self.H, qt.QobjEvo):
            H = self.H(t)
        else:
            H = self.H
        return qt.liouvillian(H=H, c_ops=Ls)


class Interaction_Picture_Input_Output:
    def __init__(self):
        self.psi0 = qt.tensor(qt.basis(N2, 20, offset=offset), qt.basis(d, 0), qt.basis(M2, 0))

        au = qt.destroy(N2, offset=offset)
        c = qt.destroy(d)
        av = qt.destroy(M2)
        Iu = qt.qeye(N2)
        Id = qt.qeye(d)
        Iv = qt.qeye(M2)
        self.au_tot = qt.tensor(au, Id, Iv)
        self.c_tot = qt.tensor(Iu, c, Iv)
        self.av_tot = qt.tensor(Iu, Id, av)
        self.I = qt.tensor(Iu, Id, Iv)

        self.u_pulse = Pulse(shape=gauss, in_going=True, args=[tp, tau])
        self.v_pulse = Pulse(shape=gauss, in_going=False, args=[tp, tau])
        self.H = qt.QobjEvo([[1j * np.sqrt(gamma) * (self.au_tot.dag() * self.c_tot - self.c_tot.dag() * self.au_tot),
                              lambda t, args: self.u_pulse.u(t)],
                             [1j * np.sqrt(gamma) * (self.av_tot.dag() * self.c_tot - self.c_tot.dag() * self.av_tot),
                              lambda t, args: self.u_pulse.u(t) * cot(2 * theta(t))]])
        self.L = [np.sqrt(gamma) * self.c_tot - qt.QobjEvo([[self.av_tot,
                                                             lambda t, args: (np.tan(theta(t)) + cot(theta(t))) * self.u_pulse.u(t)]])]

    def get_expectation_observables(self) -> Union[List[qt.Qobj], Callable]:
        return [self.au_tot.dag() * self.au_tot, self.c_tot.dag() * self.c_tot, self.av_tot.dag() * self.av_tot]

    def liouvillian(self, t: float, args) -> qt.Qobj:
        """
        Gets the liouvillian from the reduced Hamiltonian and collapse operators. This should only be used on the final
        total component for the whole SLH-network
        :return: The liouvillian ready to use for a master equation solver
        """
        Ls = [row(t) if isinstance(row, qt.QobjEvo) else row for row in self.L]
        if isinstance(self.H, qt.QobjEvo):
            H = self.H(t)
        else:
            H = self.H
        return qt.liouvillian(H=H, c_ops=Ls)


if __name__ == '__main__':
    main()
