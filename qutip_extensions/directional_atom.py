import numpy as np
import qutip as qt
from scipy import sparse as sps


def main():
    psi0, sigma1_minus, sigma2_minus, I = get_directional_atom()

    print(sigma1_minus.dag() * sigma1_minus * sigma1_minus.dag() * psi0)


def get_directional_atom():
    size = 3
    shape = (size, size)

    sigma1_minus = qt.Qobj(sps.coo_matrix(([1], ([0], [1])), shape=shape))
    sigma2_minus = qt.Qobj(sps.coo_matrix(([1], ([0], [2])), shape=shape))
    I = qt.Qobj(sps.coo_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=shape))
    psi0 = qt.basis(size, 0)

    return psi0, sigma1_minus, sigma2_minus, I


if __name__ == '__main__':
    main()
