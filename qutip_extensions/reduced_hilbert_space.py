import qutip as qt
import numpy as np
from typing import List, Tuple
from scipy import sparse as sps


def index_map(N, M_old, M_new) -> List[Tuple[int, int]]:
    """
    Maps the indexes of output operator into the corresponding input operator indexes for the next iteration
    :param N: The number of input oscillators
    :param M_old: The number of output oscillators in old system (same as number of input oscillators in new system)
    :param M_new: The number of output oscillators for new system
    :return: The mapped index
    """
    i1 = 2 + N
    j1 = 2     # The index for |1>|1...> state
    i2 = 2 + (M_old + N) + N
    j2 = 2 + M_old + M_new   # The index for |0>|1...> state
    i3 = 2 + 2*(M_old + N) + N
    j3 = 2 + 2 * (M_old + M_new)   # The index for |0>|2...> state
    i_pairs = [i1, i2, i3]
    j_pairs = [j1, j2, j3]
    return list(zip(i_pairs, j_pairs))


def create_initial_state(psi: qt.Qobj, N, M_old, M_new) -> qt.Qobj:
    """
    Creates the initial state of the system given a state vector of excitation in each oscillator in the order
    two-level-system, 1...N, 1...M. A maximum of 2 excitations are allowed for now
    :param psi: The output state vector of the 1...M output oscillators from the last iteration
    :param N: The number of input oscillators in old system
    :param M_old: The number of output oscillators in old system (same as number of input oscillators in new system)
    :param M_new: The number of output oscillators for new system
    :return: The initial state vector with all modes
    """
    if psi.isket:
        x_old = N + M_old
        Y_old = (x_old + 2) * (x_old + 3) // 2 - 1
        x_new = M_old + M_new
        Y_new = (x_new + 2) * (x_new + 3) // 2 - 1
        psi0_array = np.zeros([Y_new], dtype=np.complex_)
        psi_data = psi.data

        # Deal with atomic state:
        for i in range(2):
            psi0_array[i] = psi_data[i, 0]

        # Deal with singly and doubly excited states of oscillators
        index_pairs = index_map(N, M_old, M_new)
        for pair in index_pairs:
            for i in range(M_old):
                psi0_array[pair[1] + i] = psi_data[pair[0] + i, 0]

        # Deal with pairs of excited states
        old = 2 + 3*(N + M_old) + sum([N + M_old - k - 1 for k in range(N)])
        new = 2 + 3*(M_old + M_new)
        for i in range(Y_old - old):
            psi0_array[new + i] = psi_data[old + i, 0]

        return qt.Qobj(psi0_array)
    else:
        return create_initial_state_dm(psi, N, M_old, M_new)


def create_initial_state_dm(rho: qt.Qobj, N, M_old, M_new) -> qt.Qobj:
    """
    Creates the initial state of the system given a density matrix of excitations in each oscillator in the order
    two-level-system, 1...N, 1...M. A maximum of 2 excitations are allowed for now
    :param rho: The output density matrix of the 1...M output oscillators from last iteration
    :param N: The number of input oscillators in old system
    :param M_old: The number of output oscillators in old system (same as number of input oscillators in new system)
    :param M_new: The number of output oscillators for new system
    :return: The initial density matrix with all modes
    """
    x_old = N + M_old
    Y_old = (x_old + 2) * (x_old + 3) // 2 - 1
    x_new = M_old + M_new
    Y_new = (x_new + 2) * (x_new + 3) // 2 - 1
    row_indexes = []
    col_indexes = []
    data = []
    shape = (Y_new, Y_new)
    #rho0_array = np.zeros([Y_new, Y_new], dtype=np.complex_)
    rho_data = rho.data

    # Deal with atomic state:
    for i in range(2):
        for j in range(2):
            row_indexes.append(i)
            col_indexes.append(j)
            data.append(rho_data[i, j])
            #rho0_array[i, j] = rho_data[i, j]

    # Deal with singly and doubly excited states of oscillators
    index_pairs = index_map(N, M_old, M_new)
    for pair in index_pairs:
        for i in range(M_old):
            for j in range(M_old):
                row_indexes.append(pair[1] + i)
                col_indexes.append(pair[1] + j)
                data.append(rho_data[pair[0] + i, pair[0] + j])
                #rho0_array[pair[1] + i, pair[1] + j] = rho_data[pair[0] + i, pair[0] + j]

    # Deal with pairs of excited states
    old = 2 + 3*(N + M_old) + sum([N + M_old - k - 1 for k in range(N)])
    new = 2 + 3*(M_old + M_new)
    for i in range(Y_old - old):
        for j in range(Y_old - old):
            row_indexes.append(new + i)
            col_indexes.append(new + j)
            data.append(rho_data[old + i, old + j])
            #rho0_array[new + i, new + j] = rho_data[old + i, old + j]
    rho0_array = sps.coo_matrix((data, (row_indexes, col_indexes)), shape=shape)
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
    if n == 1:
        size = N + M + 2
    elif n == 2:
        size = (N + M + 2) * (N + M + 3) // 2 - 1   # The minimal size of the Hilbert space of N + M + two level system
    else:
        raise NotImplementedError('n greater than 2 has not yet been implemented')
    shape = (size, size)
    #sps_array = sps.coo_matrix(shape)
    for i in range(N + M + 1):
        #ai_array = sps_array.copy()
        row_indexes = []
        col_indexes = []
        data = []
        if i == 0:
            # Do the loop for the two-level system ladder operator
            if n == 1:
                row_indexes.append(0)
                col_indexes.append(1)
                data.append(1)
            else:
                for j in range(N + M + 1):
                    if j == 0:
                        #ai_array = ai_array + sps.coo_matrix((1, (0, 1)), shape=shape)
                        row_indexes.append(0)
                        col_indexes.append(1)
                        data.append(1)
                    else:
                        # Skip the first N + M + 1 columns which are the 0 state and all the excited two level states
                        row = j + N + M + 1
                        col = j + 1
                        #ai_array = ai_array + sps.coo_matrix((1, (row, col)), shape=shape)
                        row_indexes.append(row)
                        col_indexes.append(col)
                        data.append(1)
        else:
            if n == 1:
                row_indexes.append(0)
                col_indexes.append(1 + i)
                data.append(1)
            else:
                # Do the loop for all other operators
                # Lower all the singly excited states of the oscillator
                #ai_array = ai_array + sps.coo_matrix((1, (0, N + M + 1 + i)), shape=shape)
                row_indexes.append(0)
                col_indexes.append(N + M + 1 + i)
                data.append(1)
                #ai_array = ai_array + sps.coo_matrix((1, (1, i + 1)), shape=shape)
                row_indexes.append(1)
                col_indexes.append(i + 1)
                data.append(1)
                # Lower all the double excited states of the oscillator
                #ai_array = ai_array + sps.coo_matrix((np.sqrt(2), (N + M + 1 + i, 2*(N + M) + 1 + i)), shape=shape)
                row_indexes.append(N + M + 1 + i)
                col_indexes.append(2*(N + M) + 1 + i)
                data.append(np.sqrt(2))

                # Now lower all pairs of singly excited states of the oscillator
                for j in range(0, i - 1):
                    # All operators before number i
                    row = 2 + (N + M) + j
                    col = 1 + 3 * (N + M) + sum([N + M - k - 1 for k in range(j)]) + i - (j + 1)
                    #ai_array = ai_array + sps.coo_matrix((1, (row, col)), shape=shape)
                    row_indexes.append(row)
                    col_indexes.append(col)
                    data.append(1)
                for j in range(0, N + M - i):
                    # All operators after number i
                    row = 2 + (N + M) + i + j
                    col = 2 + 3*(N + M) + sum([N + M - k - 1 for k in range(i - 1)]) + j
                    #ai_array = ai_array + sps.coo_matrix((1, (row, col)), shape=shape)
                    row_indexes.append(row)
                    col_indexes.append(col)
                    data.append(1)
        ai_array = sps.coo_matrix((data, (row_indexes, col_indexes)), shape=shape)
        out_operators.append(qt.Qobj(ai_array))
    return out_operators
