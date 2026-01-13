import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.interpolate import CubicSpline

from SLH import network as nw
from SLH import component_factory as cf
from quantumsystem import QuantumSystem
from plots import LineOptions
import pulse as p
import physics_functions as ph

from typing import Union, List, Callable, Any


D = 2

gamma = 1
Omega = 1
Delta = 1
T = 10
nT = 1000


def main():
    tlist = np.linspace(0, T, nT)
    driven_atom = DrivenAtom(tlist, Omega, w0=Delta)
    vals_in_arms, vecs_in_arms = ph.run_takagi(driven_atom, n=2, trim=False, plot=False)
    vals, vecs = vals_in_arms[0], vecs_in_arms[0]
    vals_in_arms, vecs_in_arms = ph.run_autocorrelation(driven_atom, n=2, trim=False, plot=False)
    vals2, vecs2 = vals_in_arms[0], vecs_in_arms[0]

    plt.figure()
    plt.plot(tlist, vecs[0])
    plt.plot(tlist, vecs2[0], '--')
    plt.plot(tlist, vecs[1])
    plt.plot(tlist, vecs2[1], '--')
    #plt.plot(tlist, vecs[2])
    #plt.plot(tlist, -vecs_old[2], '--')
    plt.show()

    driven_atom_with_output = DrivenAtomWithOutput(vecs, tlist, Omega, Delta, [D, 12, 8], 11)
    result: qt.solver.Result = ph.run_quantum_system(driven_atom_with_output, plot=False, verbose='enhanced')

    for i, v in enumerate(vecs):
        print(vals[i], result.expect[i + 1][-1])


class DrivenAtom(QuantumSystem):
    def __init__(self, tlist, Omega, w0, psi0=None):
        c = qt.destroy(D)
        Id = qt.qeye(D)
        self.c1 = c
        self.I = Id
        self.Omega = Omega
        self.w0 = w0
        if psi0 is None:
            psi0 = qt.basis(D, 0)
        pulses = []
        super().__init__(psi0, pulses, tlist)

    def create_component(self) -> nw.Component:
        driven_atom = nw.Component(S=nw.MatrixOperator(self.I), L=nw.MatrixOperator(np.sqrt(gamma) * self.c1),
                                   H=self.Omega * self.c1.dag() + np.conjugate(self.Omega) * self.c1
                                     + self.w0 * self.c1.dag() * self.c1)
        return driven_atom

    def get_expectation_observables(self) -> Union[List[qt.Qobj], Callable]:
        return [self.c1.dag() * self.c1]

    def get_plotting_options(self) -> Any:
        pulse_options = []
        content_options = [LineOptions(linetype='-', linewidth=4, color='r',
                                       label=r'$\langle \hat{c}_{1}^\dagger\hat{c}_{1} \rangle$')
                           ]
        return pulse_options, content_options


class DrivenAtomWithOutput(QuantumSystem):
    def __init__(self, us: List[np.ndarray], tlist, Omega, w0, Ns, excitations):
        M = len(us)
        vs, gvs = p.transform_pulses(us, tlist, is_input=False)
        self.vs = [p.Pulse('undefined', in_going=False,
                           args=[CubicSpline(tlist, vs[i]), CubicSpline(tlist, gvs[i])]) for i in range(M)]
        a_list = qt.enr_destroy(Ns, excitations)
        self.c = a_list[0]
        self.a_list = a_list[1:]
        self.I = qt.enr_identity(Ns, excitations)
        self.Omega = Omega
        self.w0 = w0
        psi0 = qt.enr_fock(Ns, excitations, [0 for _ in Ns])
        pulses = []
        super().__init__(psi0, pulses, tlist)

    def create_component(self) -> nw.Component:
        c1 = self.c
        driven_atom = nw.Component(S=nw.MatrixOperator(self.I), L=nw.MatrixOperator(np.sqrt(gamma) * c1),
                                   H=self.Omega * c1.dag() + np.conjugate(self.Omega) * c1 + self.w0 * c1.dag() * c1)
        total_component = driven_atom
        for i, v in enumerate(self.vs):
            cavity = cf.create_cavity(self.I, self.a_list[i], v.g, 0)
            total_component = nw.series_product(total_component, cavity)
        return total_component

    def get_expectation_observables(self) -> Union[List[qt.Qobj], Callable]:
        return [self.c.dag() * self.c] + [ai * ai for ai in self.a_list]

    def get_plotting_options(self) -> Any:
        pulse_options = []
        adaga_content_options = [LineOptions(linetype='--', linewidth=4, color='r',
                                       label=r'$\langle \hat{a}_{i}^\dagger\hat{a}_{i} \rangle$') for _ in range(len(self.vs))]
        content_options = [LineOptions(linetype='-', linewidth=4, color='r',
                                       label=r'$\langle \hat{c}_{1}^\dagger\hat{c}_{1} \rangle$'),
                           *adaga_content_options]
        return pulse_options, content_options


if __name__ == '__main__':
    main()
