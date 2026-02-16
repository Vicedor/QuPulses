# Parametric Amplification
A module for computing the quantum state of the output modes of a parametric amplifier, given a
single input mode. The quantum state of the input mode can be any quantum state, also non-Gaussian
qunatum states.

The module implements the methods introduced in the manuscript {insert reference}. It builds upon
earlier work, where it was shown that the output quantum state of a single input mode is captured
by at most two modes in the output [1]. This allows an effective calculation of the output quantum
state using Multidimensional Hermite Polynomials.

The file squeezing.py contains the module, which can be readily used to compute the output quantum
states of a parametric amplifier. It also contains the procedure to create the content of figure 2
in [2] through the main-method.

The example uses an optical parametric oscillator (OPO), whose transformation matrices can be found
in the opo.py file.

Some common functionalities are included in the helper_functions.py file.

The creating-CSS-using-OPO.py file includes the code for optimizing a Coherent State Superposition,
an odd schrödinger cat state, by squeezing a single photon input state in an OPO. This file will
generate the data for figure 3 in [2].

The rest of the files are legacy files, and may eventually be removed.

[1]: [Parametric amplification of a quantum pulse, Tziperman, Christiansen, Kaminer, and Mølmer, 2024](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.110.053712)

[2]: link to be inserted
