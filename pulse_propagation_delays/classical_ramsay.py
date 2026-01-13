import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import trapezoid
from scipy.special import erf
import sys
import os
import pickle

from typing import Callable, Tuple

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


# Numerical parameters
epsilon = 1e-10                     # Kill rate to avoid numerical instability

# Pulse shapes
gauss = "gaussian"

N = 4
d = 2

n = N - 1
alpha = np.sqrt(n)
gamma = 1

if N == 4:
    t_delay = 1.4
    tp = 0.8
    T2 = 9
    T = 11
    Delta_max = 15

elif N == 10:
    t_delay = 0.5
    tp = 0.3
    T2 = 6
    T = 8
    Delta_max = 25

elif N == 31:
    t_delay = 0.2
    tp = 0.15
    T2 = 4
    T = 8
    Delta_max = 50

tau = np.pi ** (3/2) / (8 * gamma * n)
nT = 2000
cut = int(nT * T2 / T)
times = np.linspace(0, T, nT)

file = rf'classical_ramsay_with_{n}_photons.pkl'


def main():
    Deltas = np.linspace(0, Delta_max, 251)
    if os.path.exists(file):
        with open(file, 'rb') as f:
            dic = pickle.load(f)
            populations = dic['populations']
    else:
        populations = np.zeros(shape=(len(Deltas), len(times)))
        for i, D in enumerate(Deltas):
            sys.stdout.write("\r" + "Iteration " + str(i) + " out of " + str(len(Deltas)))
            result: qt.solver.Result = interaction_picture_solution(delta=D)
            populations[i, :] = result.expect[0]
            sys.stdout.flush()
        with open(file, 'wb') as f:
            pickle.dump({'n': n, 'times': times, 'gamma': gamma, 't_delay': t_delay, 'tp': tp, 'tau': tau,
                         'T2': T2, 'Deltas': Deltas, 'populations': populations, 'cut': cut}, f)

    extended_deltas = np.concatenate([np.flip(-Deltas), Deltas])
    plt.figure(figsize=(5.2, 4))
    t_max2_ind = round((t_delay + tp + 2 * tau) / T * nT)
    plt.plot(extended_deltas, np.concatenate([np.flip(populations[:, t_max2_ind]), populations[:, t_max2_ind]]), '-')
    plt.title(rf'$|n={n}\rangle$ incoming photons')
    plt.ylim([-0.01, 1.01])
    plt.xlabel(r'Detuning $\Delta$')
    plt.ylabel(r'$\rho_{22}(t_p + \tau + 2t_w)$')
    plt.tight_layout()
    plt.savefig(f'classical_atomic_excitation_{n}_photons.pdf', format='pdf', dpi=300)
    plt.show()


def analytical(deltas):
    return (np.sin(deltas * tau) / (deltas * tau)) ** 2 * np.cos(deltas * (t_delay + tau)) ** 2


def cot(x):
    return 0 if np.isclose(np.sin(x), 0) else np.cos(x) / np.sin(x)


def tan(x):
    return 0 if np.isclose(np.cos(x), 0) else np.sin(x) / np.cos(x)


def csc(x):
    return 0 if np.isclose(np.sin(x), 0) else 1 / np.sin(x)


def gaussian(tp: float, tau: float) -> Callable[[float], float]:
    """
    Returns a gaussian function with given tp and tau parameters
    :param tp: The offset in time of the gaussian
    :param tau: The width of the gaussian
    :return: A gaussian function with the given parameters
    """
    return lambda t: np.exp(-(t - tp) ** 2 / (2 * tau ** 2)) / (np.sqrt(tau) * np.pi ** 0.25)


def gaussian_squared_integral(tp: float, tau: float) -> Callable[[float], float]:
    """
    Returns a function which evaluates the integral of the square of a gaussian function given the tp and tau parameters
    :param tp: The offset in time of the gaussian
    :param tau: The width of the gaussian
    :return: A function evaluating the integral of gaussian with the given parameters
    """
    return lambda t: 0.5 * (erf((t - tp) / tau) + 1)


def interaction_picture_solution(delta: float = 0) -> qt.solver.Result:
    w1_pulse = Pulse(gauss, in_going=True, args=[tp + t_delay, tau])
    w2_pulse = Pulse(gauss, in_going=True, args=[tp, tau])

    sigma_minus: qt.Qobj = qt.destroy(d)
    sigma_plus: qt.Qobj = sigma_minus.dag()
    I: qt.Qobj = qt.qeye(d)

    c1 = alpha / np.sqrt(2)
    c2 = alpha / np.sqrt(2)

    psi0: qt.Qobj = qt.basis(d, 0)

    L1dagL1 = lambda t, state: qt.expect(L1(t).dag() * L1(t), state)
    L2dagL2 = lambda t, state: qt.expect(L2(t).dag() * L2(t), state)
    e_ops = [sigma_plus * sigma_minus, L1dagL1, L2dagL2]

    H0 = delta * sigma_plus * sigma_minus
    H1 = lambda t: 1j * np.sqrt(gamma / 2) * w1_pulse.u(t) * np.conjugate(c1) * sigma_minus
    H2 = lambda t: 1j * np.sqrt(gamma / 2) * w2_pulse.u(t) * np.conjugate(c2) * sigma_minus

    H = lambda t: H0 + H1(t) + H1(t).dag() + H2(t) + H2(t).dag()

    L1 = lambda t: np.sqrt(gamma) * sigma_minus
    L2 = lambda t: 0 * I

    return qt.mesolve(H, psi0, times, c_ops=[L1, L2], e_ops=e_ops, options={'max_step': 0.1})


class Pulse:
    def __init__(self, shape: str, in_going: bool, args):
        self.shape = shape
        self.in_going: bool = in_going
        self._u, self._g = self._get_mode_function(args)

    def set_pulse_args(self, args):
        """
        Redefines the pulse-mode with new arguments. Does not change the functional shape of the pulse, only the
        arguments for the mode function
        :param args: The new arguments
        """
        self._u, self._g = self._get_mode_function(args)

    def _get_mode_function(self, args) -> Tuple[Callable[[float], float], Callable[[float], float]]:
        """
        Gets the mode functions u and g from the shape attribute
        :return: The mode functions u and g
        """
        if self.shape == gauss:
            u = gaussian(*args)
            g = gaussian_squared_integral(*args)
        else:
            raise ValueError(self.shape + " is not a defined pulse mode.")
        return u, g

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


if __name__ == '__main__':
    main()
