#!python3
"""
In this file I test the SLH-framework I have developed by applying it to Kiilerich's pulse scheme, which should produce
the same results as the no_interaction_picture.py file in InteractionPicturePulses package.
"""
import matplotlib.pyplot as plt

from util.constants import *
import util.pulse as p
import util.physics_functions as ph
import util.plots as plots
import qutip as qt
import numpy as np
import network as nw

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})

tp = 4
tau = 1
w0 = 0
Delta = 0
gamma = 1

T = 12                              # Maximum time to run simulation (in units gamma^-1)
nT = 1000                           # The number of points in time to include
times = np.linspace(0, T, nT)       # The list of points in time to evaluate the observables

N = 2
d = 2
M = 2
offset = 0

u_shape = gaussian
v_shape = filtered_gaussian

# Initial state
# u: incoming photon pulse
psi0u = qt.basis(N, 1, offset=offset)  # Initial number of input photons
#psi0u = 0.77 * qt.basis(N, 1) + 0.54 * qt.basis(N, 3) + 0.29 * qt.basis(N, 5) + 0.14 * qt.basis(N, 7)
# s: system
psi0s = qt.basis(d, 0)  # Initial system state
# v: outgoing photon pulse
psi0v = qt.basis(M, 0)  # Initial number of output photons
psi0 = qt.tensor(psi0u, psi0s, psi0v)

u_pulse = p.Pulse(shape=u_shape, in_going=True, args=[tp, tau])#, gamma, w0, nT, times])
v_pulse = p.Pulse(shape=v_shape, in_going=False, args=[tp, tau, gamma, w0, nT, times])
pulses = [u_pulse, v_pulse]

# Plotting options
pulse_options = [({"linetype": '-', "linewidth": 4, "color": 'r', "label": "$u(t)$"},
                  {"linetype": '-', "linewidth": 4, "color": 'r', "label": "$|g_u(t)|^2$"}),
                 ({"linetype": '--', "linewidth": 4, "color": 'b', "label": "$v(t)$"},
                  {"linetype": '--', "linewidth": 4, "color": 'b', "label": "$|g_v(t)|^2$"})]
content_options = [{"linetype": '-', "linewidth": 4, "color": 'r',
                    "label": r'$\langle \hat{a}_u^\dagger\hat{a}_u \rangle$'},
                   {"linetype": '--', "linewidth": 4, "color": 'b',
                    "label": r"$\langle \hat{a}_v^\dagger\hat{a}_v \rangle$"},
                   {"linetype": ':', "linewidth": 4, "color": 'g',
                    "label": r"$\langle \hat{c}^\dagger\hat{c} \rangle$"}]


def main():
    au: qt.Qobj = qt.destroy(N)
    c: qt.Qobj = qt.destroy(d)
    av: qt.Qobj = qt.destroy(M)
    Iu: qt.Qobj = qt.qeye(N)
    Ic: qt.Qobj = qt.qeye(d)
    Iv: qt.Qobj = qt.qeye(M)
    I: qt. Qobj = qt.tensor(Iu, Ic, Iv)
    au_tot: qt.Qobj = qt.tensor(au, Ic, Iv)
    c_tot: qt.Qobj = qt.tensor(Iu, c, Iv)
    av_tot: qt.Qobj = qt.tensor(Iu, Ic, av)
    au_t: qt.QobjEvo = qt.QobjEvo([[au_tot, lambda t, args: np.conjugate(u_pulse.g(t))]])
    av_t: qt.QobjEvo = qt.QobjEvo([[av_tot, lambda t, args: np.conjugate(v_pulse.g(t))]])

    u_cavity: nw.Component = nw.Component(nw.MatrixOperator(I), nw.MatrixOperator(au_t),
                                          w0*au_t.dag()*au_t)
    c_component: nw.Component = nw.Component(nw.MatrixOperator(I), nw.MatrixOperator(np.sqrt(gamma)*c_tot),
                                             Delta*c_tot.dag()*c_tot)
    v_cavity: nw.Component = nw.Component(nw.MatrixOperator(I), nw.MatrixOperator(av_t),
                                          w0*av_t.dag()*av_t)
    total_system: nw.Component = nw.series_product(nw.series_product(u_cavity, c_component), v_cavity)

    au_dag_au = qt.tensor(au.dag() * au, Ic, Iv)  # Total number operator for incoming pulse
    av_dag_av = qt.tensor(Iu, Ic, av.dag() * av)  # Total number operator for outgoing pulse
    c_dag_c = qt.tensor(Iu, c.dag() * c, Iv)  # Total number operator for system excitations
    e_ops = [au_dag_au, av_dag_av, c_dag_c]

    # Test plotting options
    assert len(pulse_options) == len(pulses)
    assert len(e_ops) == len(content_options)

    """
    for t in times:
        assert total_system.H(t) == 0.5j * (np.sqrt(gamma) * u_pulse.g(t) * au_tot.dag() * c_tot +
                                            np.sqrt(gamma) * np.conjugate(v_pulse.g(t)) * c_tot.dag() * av_tot +
                                            u_pulse.g(t) * np.conjugate(v_pulse.g(t)) * au_tot.dag() * av_tot -
                                            np.sqrt(gamma) * np.conjugate(u_pulse.g(t)) * c_tot.dag() * au_tot -
                                            np.sqrt(gamma) * v_pulse.g(t) * av_tot.dag() * c_tot -
                                            np.conjugate(u_pulse.g(t)) * v_pulse.g(t) * av_tot.dag() * au_tot)
        assert total_system.L.convert_to_qobj()(t) == np.sqrt(gamma) * c_tot + np.conjugate(u_pulse.g(t)) * au_tot\
                                                      + np.conjugate(v_pulse.g(t)) * av_tot
    """

    result = ph.calculate_expectations_and_states(total_system, psi0, e_ops, times,
                                                  qt.Options(nsteps=1000000000, store_states=1, atol=1e-8, rtol=1e-6))
    n1, n2, Pe = result.expect
    rho = result.states

    print(qt.expect(av_dag_av, rho[-1]))

    plots.plot_system_contents(times, pulses, pulse_options, result.expect, content_options)


if __name__ == '__main__':
    main()
