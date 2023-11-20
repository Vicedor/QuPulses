import qutip as qt
import numpy as np
from typing import List, Tuple


def index_map(m: int, N, M) -> List[Tuple[int, int]]:
    """
    Maps the indexes of an output operator m into the corresponding input operator indexes for the next iteration
    :param m: The index of the output operator to map
    :param N: The number of input oscillators
    :param M: The number of output oscillators
    :return: The mapped index
    """
    i1 = 2 + (m + N)
    j1 = 2 + m     # The index for |1>|...1m...> state
    i2 = 2 + (M + N) + (m + N)
    j2 = 2 + 2 * M + m   # The index for |0>|...1m...> state
    i3 = 2 + 2*(M + N) + (m + N)
    j3 = 2 + 4 * M + m   # The index for |0>|...2m...> state
    i_pairs = [i1, i2, i3]
    j_pairs = [j1, j2, j3]
    return list(zip(i_pairs, j_pairs))


def create_initial_state(rho: qt.Qobj, N, M) -> qt.Qobj:
    """
    Creates the initial state of the system given a list of the number of excitation in each oscillator in the order
    two-level-system, 1...N, 1...M. A maximum of 2 excitations are allowed for now
    :param rho: The output density matrix of the 1...M output oscillators from last iteration
    :param N: The number of input oscillators
    :param M: The number of output oscillators
    :return: The initial density matrix with all modes
    """
    x_old = N + M
    Y_old = (x_old + 2) * (x_old + 3) // 2 - 1
    x_new = 2 * M
    Y_new = (x_new + 2) * (x_new + 3) // 2 - 1
    rho0_array = np.zeros([Y_new, Y_new], dtype=np.complex_)
    rho_data = rho.data
    print(rho)

    # Deal with atomic state:
    for i in range(2):
        for j in range(2):
            rho0_array[i, j] = rho_data[i, j]

    # Deal with singly and doubly excited states of oscillators
    for m in range(M):
        index_pairs = index_map(m, N, M)
        for pair in index_pairs:
            for i in range(M):
                for j in range(M):
                    rho0_array[pair[1] + i, pair[1] + j] = rho_data[pair[0] + i, pair[0] + j]

    # Deal with pairs of excited states
    old = 2 + 3*(N + M) + sum([N + M - k - 1 for k in range(N)])
    new = 2 + 3*(M + M)
    print(new)
    for i in range(Y_old - old):
        for j in range(Y_old - old):
            rho0_array[new + i, new + j] = rho_data[old + i, old + j]

    return qt.Qobj(rho0_array)


def get_N_plus_M_oscillator_space(N: int, M: int, n: int = 2) -> List[qt.Qobj]:
    """
    Returns the ladder operators for a Hilbert space with at most n excitations and N oscillators before a two-level
    system, and M oscillators after the two-level systems. This is used for the case of N oscillators emitting pulses
    with quantum content and M oscillators picking up the content of the output pulses

    Ordering of Hilbert space (first ket is two-level system, other ket is oscillators 1...N, 1...M):

    |0>|0...0>, |1>|0...0>, |1>|10...0>, ..., |1>|0...01>, |0>|10...0>, ..., |0>|0...01>, |0>|20...0>, ..., |0>|0...02>,
    |0>|110...0>, |0>|1010...0>, ..., |0>|10...01>, |0>|0110...0>, |0>|01010...0>, ..., |0>|010...01>, ..., |0>|0...011>

    :param N: The number of oscillators to the left of the two-level system
    :param M: The number of oscillators to the right of the two-level system
    :param n: The maximum number of excitations shared by the system (only implemented for n=2 so far)
    :return: A list of the ladder operators for the system in the order d, 1...N, 1...M, where d is the two level system
             and 1...N are the N oscillators before the two-level, and 1...M are the M oscillators after
    """
    out_operators: List[qt.Qobj] = []
    size = (N + M + 2) * (N + M + 3) // 2 - 1   # The minimal size of the Hilbert space of N + M + two level system
    zero_array: np.ndarray = np.zeros((size, size))
    for i in range(N + M + 1):
        ai_array = zero_array.copy()
        if i == 0:
            # Do the loop for the two-level system ladder operator
            for j in range(N + M + 1):
                if j == 0:
                    ai_array[0, 1] = 1
                else:
                    # Skip the first N + M + 1 columns which are the 0 state and all the excited two level states
                    row = j + N + M + 1
                    col = j + 1
                    ai_array[row, col] = 1
        else:
            # Do the loop for all other operators
            # Lower all the singly excited states of the oscillator
            ai_array[0, N + M + 1 + i] = 1
            ai_array[1, i + 1] = 1
            # Lower all the double excited states of the oscillator
            ai_array[N + M + 1 + i, 2*(N + M) + 1 + i] = np.sqrt(2)

            # Now lower all pairs of singly excited states of the oscillator
            for j in range(0, i - 1):
                # All operators before number i
                col = 1 + 3 * (N + M) + sum([N + M - k - 1 for k in range(j)]) + i - (j + 1)
                row = 2 + (N + M) + j
                ai_array[row, col] = 1
            for j in range(0, N + M - i):
                # All operators after number i
                col = 2 + 3*(N + M) + sum([N + M - k - 1 for k in range(i - 1)]) + j
                row = 2 + (N + M) + i + j
                ai_array[row, col] = 1
        out_operators.append(qt.Qobj(ai_array))
    return out_operators
