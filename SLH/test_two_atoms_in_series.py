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
import time
import network as nw

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})

tp = 4
tau = 1
w0 = 0
Delta1 = 0
Delta2 = 0
gamma1 = 1
gamma2 = 1

T = 12                              # Maximum time to run simulation (in units gamma^-1)
nT = 500                           # The number of points in time to include
times = np.linspace(0, T, nT)       # The list of points in time to evaluate the observables

N = 2
d = 2
offset = 0

u_shape = gaussian
v_shape = gaussian

# Initial state
# u: incoming photon pulse
psi0u = qt.basis(N, 1, offset=offset)  # Initial number of input photons
# s: system
psi0s1 = qt.basis(d, 0)  # Initial system state
psi0s2 = qt.basis(d, 0)

psi0 = qt.tensor(psi0u, psi0s1, psi0s2)

u_pulse = p.Pulse(shape=u_shape, in_going=True, args=[tp, tau])


def plot(n1, Pe1, Pe2):
    # Set up u(t), g_u(t) and g_v(t)
    gu_list = np.zeros(nT, dtype=np.complex_)
    u_list = np.zeros(nT)
    for k in range(0, nT):
        gu_list[k] = u_pulse.g(times[k]) ** 2  # abs(gu_t(times[k]))
        u_list[k] = np.real(u_pulse.u(times[k]))

    fig, ax4 = plt.subplots(figsize=(8, 8))  # , dpi=1600)

    plt.subplot(4, 1, 1)
    plt.plot(times, np.real(u_list), '-', linewidth=4, color='r', label="$u(t)$")
    plt.plot((0, T), (0, 0), 'k')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='on',  # ticks along the bottom edge are off
        top='on',  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.ylabel('$\mathrm{Modes}$')
    # plt.yticks((-0.8,0.0,0.8))
    # ylim((0,1))
    # legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='upper right', frameon=False, labelspacing=1.5)

    plt.subplot(4, 1, 2)
    plt.plot(times, gu_list, '-', linewidth=4, color='r', label="$|g_u(t)|^2$")
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='on',  # ticks along the bottom edge are off
        top='on',  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.ylabel('$\mathrm{Rates}\, (\gamma)$')
    plt.yticks((0, 4, 8), ('$0.0$', '$4.0$', '$8.0$'))
    plt.ylim((0, 8))
    # legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.legend(loc='center right', frameon=False)

    plt.subplot(4, 1, 3)
    plt.plot(times, np.real(n1), '-', linewidth=4, color='r', label=r'$\langle \hat{a}_u^\dagger\hat{a}_u \rangle$')
    plt.plot(times, np.real(Pe1), ':', linewidth=4, color='g', label=r"$\langle \hat{c}_1^\dagger\hat{c}_1 \rangle$")
    plt.plot(times, np.real(Pe2), '--', linewidth=4, color='b', label=r"$\langle \hat{c}_2^\dagger\hat{c}_2 \rangle$")
    plt.xlabel('$\mathrm{Time}\, (\gamma^{-1})$')
    #plt.ylim((-0.01, 1.01))
    plt.ylabel('$\mathrm{Exctiations}$')
    # plt.yticks((0, 1, 2))
    plt.grid()
    plt.legend(loc='center right', frameon=False)
    # plt.savefig('figures/empty_cavity.png', bbox_inches='tight')

    plt.show()


def integrate_master_equation(liouvillian, psi, e_ops):
    """
    Integrates the master equation for the system specifications specified in the setup.py file
    :param liouvillian: A liouvillian object containing the Hamiltonian and Lindblad operators
    :param psi: The initial state as a ket
    :param e_ops: The observables to be tracked during the time-evolution
    :return: The expectation values of the number operators for the ingoing pulse, outgoing pulse and system excitations
             in that order
    """
    if psi.isket:
        dm = qt.ket2dm(psi)  # Density matrix of initial state
    else:
        dm = psi
    output = qt.mesolve(liouvillian, dm, tlist=times, c_ops=[], e_ops=e_ops,
                        options=qt.Options(nsteps=1000000000, store_states=1, atol=1e-8, rtol=1e-6))
    return output


def main():
    au: qt.Qobj = qt.destroy(N)
    c1: qt.Qobj = qt.destroy(d)
    c2: qt.Qobj = qt.destroy(d)
    Iu: qt.Qobj = qt.qeye(N)
    Ic1: qt.Qobj = qt.qeye(d)
    Ic2: qt.Qobj = qt.qeye(d)
    I: qt. Qobj = qt.tensor(Iu, Ic1, Ic2)
    au_tot: qt.Qobj = qt.tensor(au, Ic1, Ic2)
    c1_tot: qt.Qobj = qt.tensor(Iu, c1, Ic2)
    c2_tot: qt.Qobj = qt.tensor(Iu, Ic1, c2)
    au_t: qt.QobjEvo = qt.QobjEvo([[au_tot, lambda t, args: np.conjugate(u_pulse.g(t))]])

    u_cavity: nw.Component = nw.Component(nw.MatrixOperator(I), nw.MatrixOperator(au_t),
                                          w0*au_t.dag()*au_t)
    c1_component: nw.Component = nw.Component(nw.MatrixOperator(I), nw.MatrixOperator(np.sqrt(gamma1)*c1_tot),
                                              Delta1*c1_tot.dag()*c1_tot)
    c2_component: nw.Component = nw.Component(nw.MatrixOperator(I), nw.MatrixOperator(np.sqrt(gamma2)*c2_tot),
                                              Delta2*c2_tot.dag()*c2_tot)
    total_system: nw.Component = nw.series_product(nw.series_product(u_cavity, c1_component), c2_component)

    au_dag_au = qt.tensor(au.dag() * au, Ic1, Ic2)  # Total number operator for incoming pulse
    c1_dag_c1 = qt.tensor(Iu, c1.dag() * c1, Ic2)  # Total number operator for system excitations
    c2_dag_c2 = qt.tensor(Iu, Ic1, c2.dag() * c2)  # Total number operator for outgoing pulse
    e_ops = [au_dag_au, c1_dag_c1, c2_dag_c2]

    print("Initializing simulation")
    t1 = time.time()
    result = integrate_master_equation(total_system.liouvillian, psi0, e_ops)
    print(f"Finished in {time.time() - t1} seconds!")
    n1, Pe1, Pe2 = result.expect
    rho = result.states

    plot(n1, Pe1, Pe2)

    autocorr_mat, vals, vecs = ph.get_most_populated_modes(total_system.liouvillian, total_system.get_Ls()[0],
                                                           psi0, times, 2)

    plots.plot_autocorrelation(autocorr_mat=autocorr_mat, vs=vecs, eigs=vals, times=times)


if __name__ == '__main__':
    main()
