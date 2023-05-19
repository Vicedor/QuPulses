import qutip as qt
from typing import List, Union, Optional


class MatrixOperator:
    """
    A matrix of operators
    """

    def __init__(self, array: Union[complex, qt.Qobj, qt.QobjEvo, List[List[Union[complex, qt.Qobj, qt.QobjEvo]]]]):
        if not isinstance(array, List):
            array = [[array]]
        self.array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = array
        self.m: int = len(array)
        self.n: int = len(array[0])
        for row in array:
            if len(row) != self.n:
                raise AssertionError("Each row in the matrix operator must have the same length")

    def dag(self) -> "MatrixOperator":
        """
        Performs the dagger operation on the matrix operator as eq. 51b in SLH framework (Combes)
        """
        m: MatrixOperator = self.conjugate()
        return m.transpose()

    def transpose(self) -> "MatrixOperator":
        """
        Performs the transpose operation on the matrix operator as eq. 51b in SLH framework (Combes)
        """
        new_array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = [[0 for _ in range(self.m)] for _ in range(self.n)]
        for i, row in enumerate(self.array):
            for j, operator in enumerate(row):
                new_array[j][i] = operator
        return MatrixOperator(new_array)

    def conjugate(self) -> "MatrixOperator":
        """
        Performs the conjugation operation on the matrix operator as eq. 51b in SLH framework (Combes)
        """
        new_array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = [[0 for _ in range(self.n)] for _ in range(self.m)]
        for i, row in enumerate(self.array):
            for j, operator in enumerate(row):
                if isinstance(operator, (qt.Qobj, qt.QobjEvo)):
                    new_op = operator.dag()
                else:
                    new_op = operator.conjugate()
                new_array[i][j] = new_op
        return MatrixOperator(new_array)

    def remove(self, row: Optional[int] = None, col: Optional[int] = None) -> "MatrixOperator":
        """
        Removes the given row and the given column from the matrix operator and returns the new matrix operator
        :param row: The optional index of the row to remove
        :param col: The optional index of the column to remove
        :return: A new matrix operator with the given row and/or column removed
        """
        if row is not None:
            new_operator: MatrixOperator = self._remove_row(row=row)
        else:
            new_operator: MatrixOperator = self
        if col is not None:
            return new_operator._remove_col(col=col)
        else:
            return new_operator

    def _remove_row(self, row: int) -> "MatrixOperator":
        """
        Removes the given row
        :param row: The index of the row to remove
        :return: A new matrix operator with the given row removed
        """
        new_array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = [[0 for _ in range(self.n)] for _ in range(self.m - 1)]
        for i, r in enumerate(self.array):
            if i == row:
                continue
            else:
                if i > row:
                    i = i - 1
                new_array[i] = r
        return MatrixOperator(new_array)

    def _remove_col(self, col: int) -> "MatrixOperator":
        """
        Removes the given column
        :param col: The index of the column to remove
        :return: A new matrix operator with the given column removed
        """
        new_array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = [[0 for _ in range(self.n - 1)] for _ in range(self.m)]
        for i, row in enumerate(self.array):
            for j, op in enumerate(row):
                if j == col:
                    continue
                else:
                    if j > col:
                        j = j - 1
                    new_array[i][j] = op
        return MatrixOperator(new_array)

    def get(self, row: Optional[int] = None, col: Optional[int] = None) -> "MatrixOperator":
        """
        Returns array of the given row, col specified. Can also retrieve a whole row or a whole column if only a row or
        a column is specified
        :param row: The optional row to fetch
        :param col: The optional column to fetch
        :return: A matrix operator with only the given row and/or column
        """
        if row is not None:
            new_operator: MatrixOperator = self._get_row(row=row)
        else:
            new_operator: MatrixOperator = self
        if col is not None:
            return new_operator._get_col(col=col)
        else:
            return new_operator

    def _get_row(self, row: int) -> "MatrixOperator":
        """
        Gets a matrix operator with only the specified row
        :param row: The row to get
        :return: The matrix operator with only that row
        """
        new_array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = [[0 for _ in range(self.n)]]
        for j, op in enumerate(self.array[row]):
            new_array[0][j] = op
        return MatrixOperator(new_array)

    def _get_col(self, col: int) -> "MatrixOperator":
        """
        Gets a matrix operator with only the specified column
        :param col: The column to get
        :return: A matrix operator with only that column
        """
        new_array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = [[0] for _ in range(self.m)]
        for i, r in enumerate(self.array):
            new_array[i][0] = r[col]
        return MatrixOperator(new_array)

    def convert_to_qobj(self) -> Union[qt.Qobj, qt.QobjEvo]:
        """
        Checks if matrix operator only has one entry, and if so it returns this as a Qobj
        :return: The Qobj of the matrix operators singular entry
        """
        if len(self.array) == 1 and len(self.array[0]) == 1:
            op = self.array[0][0]
            if not isinstance(op, (qt.Qobj, qt.QobjEvo)):
                op = qt.Qobj(op)
            return op
        else:
            raise IndexError("Trying to convert matrix operator with more than one entry to Qobj")

    def __add__(self, other: Union[qt.Qobj, qt.QobjEvo, "MatrixOperator"]) -> Union[qt.Qobj, qt.QobjEvo, "MatrixOperator"]:
        """
        Adds two matrix operators as matrix addition
        :param other: The other matrix in the addition
        :return: The addition of the two matrix operators as a matrix operator
        """
        if isinstance(other, (qt.Qobj, qt.QobjEvo)):
            if len(self.array) == 1 and len(self.array[0]) == 1:
                return self.array[0][0] + other
            else:
                raise TypeError("It is not possible to add a Qobj to a Matrix Operator with more than one element")
        if self.m != other.m or self.n != other.n:
            raise IndexError("Dimensions of the two arrays added must be the same")
        new_array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = [[0 for _ in range(self.n)] for _ in range(self.m)]
        for i, row in enumerate(self.array):
            for j, col in enumerate(row):
                new_array[i][j] = col + other.array[i][j]
        return MatrixOperator(new_array)

    def __radd__(self, other: Union[qt.Qobj, qt.QobjEvo, "MatrixOperator"]) -> Union[qt.Qobj, qt.QobjEvo, "MatrixOperator"]:
        """
        Adds two matrix operators as matrix addition (or Qobj if Matrix operator only has one entry
        :param other: The other matrix in the addition
        :return: The addition of the two matrix operators as a matrix operator (or Qobj if matrix operator only has one
        entry)
        """
        return self + other

    def __neg__(self) -> "MatrixOperator":
        """
        Computes the negative of the matrix operator by negating each entry
        :return: The negative of the matrix operator
        """
        new_array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = [[0 for _ in range(self.n)] for _ in range(self.m)]
        for i, row in enumerate(self.array):
            for j, col in enumerate(row):
                new_array[i][j] = - col
        return MatrixOperator(new_array)

    def __sub__(self, other: Union[qt.Qobj, qt.QobjEvo, "MatrixOperator"]) -> Union[qt.Qobj, qt.QobjEvo, "MatrixOperator"]:
        """
        Subtracts two matrix operators as matrix addition (or Qobj if matrix operator only has one entry)
        :param other: The other matrix (or Qobj) in the subtraction
        :return: The subtraction of the two matrix operators as a matrix operator (or Qobj if matrix operator only has
        one entry)
        """
        return self + (- other)

    def __rsub__(self, other: Union[qt.Qobj, qt.QobjEvo, "MatrixOperator"]) -> Union[qt.Qobj, qt.QobjEvo, "MatrixOperator"]:
        """
        Subtracts two matrix operators as matrix addition (or Qobj if matrix operator only has one entry)
        :param other: The other matrix (or Qobj) in the subtraction
        :return: The subtraction of the two matrix operators as a matrix operator (or Qobj if matrix operator only has
        one entry)
        """
        return self - other

    def __mul__(self, other: Union[complex, qt.Qobj, qt.QobjEvo, "MatrixOperator"]) -> "MatrixOperator":
        """
        Matrix multiplication self * other with notation as in https://www.wikiwand.com/en/Matrix_multiplication
        :param other: The other matrix operator to multiply with or a complex number
        :return: The multiplication result
        """
        if isinstance(other, MatrixOperator):
            if self.n != other.m:
                raise IndexError("The matrix multiplication must match column dimension to row dimension")
            new_array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = [[0 for _ in range(other.n)] for _ in range(self.m)]
            for i in range(len(new_array)):
                for j in range(len(new_array[0])):
                    for k in range(self.n):
                        new_array[i][j] += self.array[i][k] * other.array[k][j]
        else:
            new_array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = [[0 for _ in range(self.n)] for _ in range(self.m)]
            for i, row in enumerate(self.array):
                for j, col in enumerate(row):
                    new_array[i][j] = col * other
        return MatrixOperator(new_array)

    def __rmul__(self, other: Union[complex, qt.Qobj, qt.QobjEvo, "MatrixOperator"]) -> "MatrixOperator":
        """
        Matrix multiplication other * self with notation as in https://www.wikiwand.com/en/Matrix_multiplication
        :param other: The other matrix operator to multiply with or a complex number
        :return: The multiplication result
        """
        if isinstance(other, MatrixOperator):
            if self.n != other.m:
                raise IndexError("The matrix multiplication must match column dimension to row dimension")
            new_array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = [[0 for _ in range(self.n)] for _ in
                                                                          range(other.m)]
            for i in range(len(new_array)):
                for j in range(len(new_array[0])):
                    for k in range(self.n):
                        new_array[i][j] += other.array[i][k] * self.array[k][j]
        else:
            new_array: List[List[Union[complex, qt.Qobj, qt.QobjEvo]]] = [[0 for _ in range(self.n)] for _ in
                                                                          range(self.m)]
            for i, row in enumerate(self.array):
                for j, col in enumerate(row):
                    new_array[i][j] = other * col
        return MatrixOperator(new_array)

    def __matmul__(self, other: "MatrixOperator") -> "MatrixOperator":
        return self * other

    def __repr__(self):
        return str(self.array)

    def __eq__(self, other: "MatrixOperator") -> bool:
        if len(self.array) != len(other.array):
            return False
        elif len(self.array[0]) != len(other.array[0]):
            return False
        else:
            for i, row in enumerate(self.array):
                for j, op1 in enumerate(row):
                    op2: qt.Qobj = other.array[i][j]
                    if isinstance(op1, (qt.Qobj, qt.QobjEvo)) and isinstance(op2, (qt.Qobj, qt.QobjEvo)):
                        if qt.isequal(op1, op2):
                            continue
                        else:
                            return False
                    elif isinstance(op1, (qt.Qobj, qt.QobjEvo)) or isinstance(op2, (qt.Qobj, qt.QobjEvo)):
                        return False
                    else:
                        if op1 != op2:
                            return False
        return True


class Component:
    """
    A component for the SLH-Network
    """

    def __init__(self, S: MatrixOperator, L: MatrixOperator, H: Union[complex, qt.Qobj, qt.QobjEvo]):
        """
        Initializes a component with the given SLH triple
        :param S: The scattering matrix operator
        :param L: The jump vector operator
        :param H: The Hamiltonian operator for the component
        """
        self.S: MatrixOperator = S
        self.L: MatrixOperator = L
        self.H: qt.Qobj = H

    def feedback_reduction(self, x: int, y: int) -> 'Component':
        """
        Performs the feedback reduction rule, connecting an output of this component to an input of the component
        Eq. 61 - 62 in SLH framework (Combes)
        The array in the matrix operator is 0-indexed
        :param x: The number of the output channel to connect to an input channel
        :param y: The number of the input channel to connect with the output channel
        :return: The reduced component
        """
        # Create the S matrix operator with x'th row and y'th column removed
        S_xbar_ybar: MatrixOperator = self.S.remove(row=x, col=y)
        # Create the S matrix operator with only y'th column
        S_y: MatrixOperator = self.S.get(col=y)
        # Create the S matrix operator with only the y'th column and the x'th row removed
        S_xbar_y: MatrixOperator = S_y.remove(row=x)
        # Create the S matrix operator with only the x'th row and the y'th column removed
        S_x_ybar: MatrixOperator = self.S.get(row=x).remove(col=y)
        # Create the S matrix operator with only x,y entry:
        S_xy: Union[qt.Qobj, qt.QobjEvo] = self.S.get(row=x, col=y).convert_to_qobj()
        # Create the L matrix operator with x'th row removed
        L_xbar: MatrixOperator = self.L.remove(row=x)
        L_x: Union[qt.Qobj, qt.QobjEvo] = self.L.get(row=x).convert_to_qobj()

        # Create the inverse operator
        I: qt.Qobj = qt.qeye(S_xy.dims[0])
        if I - S_xy == I*0:
            print("Feedback reduction, division by 0 error")
            I_S_xy_op: Union[qt.Qobj, qt.QobjEvo] = qt.Qobj(0)
            I_S_xy_dag_op: Union[qt.Qobj, qt.QobjEvo] = qt.Qobj(0)
        else:
            I_S_xy_op: Union[qt.Qobj, qt.QobjEvo] = (I - S_xy).inv()
            I_S_xy_dag_op: Union[qt.Qobj, qt.QobjEvo] = (I - S_xy.dag()).inv()

        # Create the reduced S matrix operator
        S_red: MatrixOperator = S_xbar_ybar + S_xbar_y * I_S_xy_op * S_x_ybar
        # Create the reduced L matrix operator
        L_red: MatrixOperator = L_xbar + S_xbar_y * I_S_xy_op * L_x
        # Create the reduced H operator
        H_red: Union[qt.Qobj, qt.QobjEvo] = self.H - 0.5j * ((self.L.dag() * S_y).convert_to_qobj() * I_S_xy_op * L_x -
                                                             L_x.dag() * I_S_xy_dag_op * (S_y.dag() * self.L).convert_to_qobj())
        return Component(S_red, L_red, H_red)

    def get_Ls(self) -> List[Union[qt.Qobj, qt.QobjEvo]]:
        """
        Gets the Lindblad operators as a list
        :return: The Lindblad operators as a list
        """
        all_zero = True
        for row in self.L.array:
            if row[0] != 0:
                all_zero = False
        if all_zero:
            return []
        else:
            return [row[0] for row in self.L.array]

    def is_L_temp_dep(self) -> bool:
        """
        Checks whether any of the Loss operators are time-dependent
        :return: boolean of whether any of the L's are time-dependent
        """
        for L in self.get_Ls():
            if isinstance(L, qt.QobjEvo):
                return True
        return False

    def liouvillian(self, t: float, args) -> qt.Qobj:
        """
        Gets the liouvillian from the reduced Hamiltonian and collapse operators. This should only be used on the final
        total component for the whole SLH-network
        :return: The liouvillian ready to use for a master equation solver
        """
        Ls = [row[0](t) if isinstance(row[0], qt.QobjEvo) else row[0] for row in self.L.array]
        if isinstance(self.H, qt.QobjEvo):
            H = self.H(t)
        else:
            H = self.H
        return qt.liouvillian(H=H, c_ops=Ls)


def series_product(component1: Component, component2: Component) -> Component:
    """
    Performs the series product to cascade output from component G1 to input to component G2.
    Keep in mind G1 <| G2 =/= G2 <| G1
    Eq. 58 in SLH framework review (Combes)
    :param component1: The left component in the series product
    :param component2: The right component in the series product
    :return: The combined component
    """
    S_tot: MatrixOperator = component2.S * component1.S
    L_tot: MatrixOperator = component2.L + component2.S * component1.L
    H_tot: qt.Qobj = component1.H + component2.H - 0.5j * (component2.L.dag() * component2.S * component1.L -
                                                           component1.L.dag() * component2.S.dag() * component2.L).convert_to_qobj()
    return Component(S_tot, L_tot, H_tot)


def concatenation_product(component1: Component, component2: Component) -> Component:
    """
    Performs the concatenation product to add two components in parallel G1 square G2
    :param component1: The top component in the parallel grouping of components
    :param component2: The bottom component in the parallel grouping of components
    :return: The combined component
    """
    return direct_coupling(component1, component2)


def direct_coupling(component1: Component, component2: Component, H_int: Optional[qt.Qobj] = None) -> Component:
    """
    Adds a direct coupling between to components in parallel. An extension to the concatenation product
    :param component1: The top component in the parallel grouping of components
    :param component2: The bottom component in the parallel grouping of components
    :param H_int: The interaction Hamiltonian between the two components
    :return: The combined component
    """
    # Create the total S operator as diagonal matrix
    S1: MatrixOperator = component1.S
    S2: MatrixOperator = component2.S
    S_tot_array: List[List[Union[complex, qt.Qobj]]] = [[0 for _ in range(S1.n + S2.n)] for _ in range(S1.m + S2.m)]
    for i, row in enumerate(S1.array):
        for j, op in enumerate(row):
            S_tot_array[i][j] = op
    for i, row in enumerate(S2.array):
        for j, op in enumerate(row):
            S_tot_array[i + S1.m][j + S1.n] = op
    S_tot: MatrixOperator = MatrixOperator(S_tot_array)

    # Create the total L operator as vector
    L1: MatrixOperator = component1.L
    L2: MatrixOperator = component2.L
    L_tot_array: List[List[complex, qt.Qobj]] = [[0] for _ in range(L1.m + L2.m)]
    for i, row in enumerate(L1.array):
        for j, op in enumerate(row):
            L_tot_array[i][j] = op
    for i, row in enumerate(L2.array):
        for j, op in enumerate(row):
            L_tot_array[i + L1.m][j] = op
    L_tot: MatrixOperator = MatrixOperator(L_tot_array)

    # Create the total H operator as QObj
    H_tot = component1.H + component2.H
    if H_int is not None:
        H_tot += H_int
    return Component(S_tot, L_tot, H_tot)


def _get_padding_element(n: int) -> Component:
    # Create n x n identity matrix
    In_array: List[List[int]] = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    In: MatrixOperator = MatrixOperator(In_array)

    # Create empty L of right dimension
    L0_array: List[List[int]] = [[0] for _ in range(n)]
    L0: MatrixOperator = MatrixOperator(L0_array)

    # Create the padding component
    I_component: Component = Component(In, L0, 0)
    return I_component


def padding_top(n: int, component: Component) -> Component:
    """
    Adds a padding of size n on top of the given component
    :param n: The number of padding channels to add to the component
    :param component: The component to add the padding to
    :return: The padded component
    """
    I_component: Component = _get_padding_element(n)

    # Add padding using concatenation product
    return concatenation_product(I_component, component)


def padding_bottom(n: int, component: Component) -> Component:
    """
    Adds a padding of size n below the given component
    :param n: The number of padding channels to add to the component
    :param component: The component to add the padding to
    :return: The padded component
    """
    I_component: Component = _get_padding_element(n)

    # Add padding using concatenation product
    return concatenation_product(component, I_component)
