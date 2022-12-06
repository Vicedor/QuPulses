#!python3

import qutip as qt
import numpy as np
import SLH.network as nw


def test_series_product():
    """
    Tests the series product using example V.2 from SLH composition rules in Combes (2017)
    """
    I1: qt.Qobj = qt.qeye(2)
    a: qt.Qobj = qt.destroy(2)
    a1: qt.Qobj = qt.tensor(a, I1)
    a2: qt.Qobj = qt.tensor(I1, a)
    I: qt.Qobj = qt.tensor(I1, I1)
    Delta1: float = 1
    Delta2: float = 2
    gamma1: float = 3
    gamma2: float = 4
    cavity1: nw.Component = nw.Component(nw.MatrixOperator(I),
                                         nw.MatrixOperator(np.sqrt(gamma1) * a1),
                                         Delta1*a1.dag()*a1)
    cavity2: nw.Component = nw.Component(nw.MatrixOperator(I),
                                         nw.MatrixOperator(np.sqrt(gamma2) * a2),
                                         Delta2 * a2.dag() * a2)
    series_component: nw.Component = nw.series_product(cavity1, cavity2)
    assert qt.isequal(series_component.S.convert_to_qobj(), I)
    assert qt.isequal(series_component.L.convert_to_qobj(), np.sqrt(gamma1)*a1 + np.sqrt(gamma2)*a2)
    assert qt.isequal(series_component.H, Delta1*a1.dag()*a1 + Delta2*a2.dag()*a2
                      - 0.5j*np.sqrt(gamma1*gamma2) * (a2.dag()*a1 - a1.dag()*a2))


def test_concatenation_product():
    """
    Tests the concatenation product using example V.2 from SLH composition rules in Combes (2017)
    """
    I1: qt.Qobj = qt.qeye(2)
    a: qt.Qobj = qt.destroy(2)
    a1: qt.Qobj = qt.tensor(a, I1)
    a2: qt.Qobj = qt.tensor(I1, a)
    I: qt.Qobj = qt.tensor(I1, I1)
    Delta1: float = 1
    Delta2: float = 2
    gamma1: float = 3
    gamma2: float = 4
    cavity1: nw.Component = nw.Component(nw.MatrixOperator(I),
                                         nw.MatrixOperator(np.sqrt(gamma1) * a1),
                                         Delta1 * a1.dag() * a1)
    cavity2: nw.Component = nw.Component(nw.MatrixOperator(I),
                                         nw.MatrixOperator(np.sqrt(gamma2) * a2),
                                         Delta2 * a2.dag() * a2)
    concatenation_component: nw.Component = nw.concatenation_product(cavity1, cavity2)
    assert concatenation_component.S == nw.MatrixOperator([[I, 0], [0, I]])
    assert concatenation_component.L == nw.MatrixOperator([[np.sqrt(gamma1)*a1], [np.sqrt(gamma2)*a2]])
    assert qt.isequal(concatenation_component.H, Delta1*a1.dag()*a1 + Delta2*a2.dag()*a2)


def test_direct_coupling():
    """
    Tests the direct coupling using example V.2 from SLH composition rules in Combes (2017)
    """
    I1: qt.Qobj = qt.qeye(2)
    a: qt.Qobj = qt.destroy(2)
    a1: qt.Qobj = qt.tensor(a, I1)
    a2: qt.Qobj = qt.tensor(I1, a)
    I: qt.Qobj = qt.tensor(I1, I1)
    Delta1: float = 1
    Delta2: float = 2
    gamma1: float = 3
    gamma2: float = 4
    xi: float = 5
    cavity1: nw.Component = nw.Component(nw.MatrixOperator(I),
                                         nw.MatrixOperator(np.sqrt(gamma1) * a1),
                                         Delta1 * a1.dag() * a1)
    cavity2: nw.Component = nw.Component(nw.MatrixOperator(I),
                                         nw.MatrixOperator(np.sqrt(gamma2) * a2),
                                         Delta2 * a2.dag() * a2)
    coupling_component: nw.Component = nw.direct_coupling(cavity1, cavity2, xi*a1.dag()*a1*a2.dag()*a2)
    assert coupling_component.S == nw.MatrixOperator([[I, 0], [0, I]])
    assert coupling_component.L == nw.MatrixOperator([[np.sqrt(gamma1) * a1], [np.sqrt(gamma2) * a2]])
    assert qt.isequal(coupling_component.H, Delta1 * a1.dag() * a1 + Delta2 * a2.dag() * a2 + xi*a1.dag()*a1*a2.dag()*a2)


def test_feedback():
    """
    Tests the feedback connection using example V.2 from SLH composition rules in Combes (2017)
    Beware: there may be an error in a sign in eq. 79 (or equivalently in eq. 61)
    """
    I: qt.Qobj = qt.qeye(2)
    a: qt.Qobj = qt.destroy(2)
    Delta: float = 1
    gamma1: float = 2
    gamma2: float = 3
    component: nw.Component = nw.Component(nw.MatrixOperator([[I, 0], [0, I]]),
                                           nw.MatrixOperator([[np.sqrt(gamma1)*a], [1j*np.sqrt(gamma2)*a]]),
                                           Delta*a.dag()*a)
    reduced_component: nw.Component = component.feedback_reduction(0, 1)
    assert qt.isequal(reduced_component.S.convert_to_qobj(), I)
    assert qt.isequal(reduced_component.L.convert_to_qobj(), (np.sqrt(gamma1) + 1j*np.sqrt(gamma2))*a)
    assert qt.isequal(reduced_component.H, (Delta - np.sqrt(gamma1*gamma2))*a.dag()*a)


def main():
    test_series_product()
    test_concatenation_product()
    test_direct_coupling()
    test_feedback()


if __name__ == '__main__':
    main()
