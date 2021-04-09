import numpy as np
from scipy import linalg as la
from multiprocessing import Pool


def chain_hamiltonian(N, J, hz, hx):
    """
    Generate the hamiltonian for an Ising chain.

    Parameters
    ----------
    N : int
        The number of qubits in the chain.
    J : float
        The nearest neighbor coupling energy.
    hz : float
        Energy of spins polarized along z axis.
    hx : float
        Energy of spins polarized along x axis.

    Returns
    -------
    H : 2D array
        Hamiltonian matrix.
    """
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])  # Unused.
    sz = np.array([[1, 0], [0, -1]])
    I = np.array([[1, 0], [0, 1]])
    H1 = np.zeros((2 ** N, 2 ** N))
    H2 = np.zeros((2 ** N, 2 ** N))
    H3 = np.zeros((2 ** N, 2 ** N))
    for i in range(N - 1):
        siz_1 = 1
        for j in range(N - 1):
            if i == j:
                siz_1 = np.kron(siz_1, sz)
                siz_1 = np.kron(siz_1, sz)
            else:
                siz_1 = np.kron(siz_1, I)
        H1 = H1 - J * siz_1
    for i in range(N):
        siz = 1
        six = 1
        for j in range(N):
            if i == j:
                siz = np.kron(siz, sz)
                six = np.kron(six, sx)
            else:
                siz = np.kron(siz, I)
                six = np.kron(six, I)
        H2 = H2 - hz * siz
        H3 = H3 - hx * six
    H = H1 + H2 + H3
    return H


def all_up(N):
    """
    Generate state for Ising chain with all spins up.

    Parameters
    ----------
    N : int
        Number of qubits in chain.

    Returns
    -------
    psi : 2D array
        State vector.
    """
    psi = 1
    for i in range(N):
        psi = np.kron(psi, np.array([[0], [1]]))
    return psi


def first_down(N):
    """
    Generate state for Ising chain with all spins up except the first.

    Parameters
    ----------
    N : int
        Number of qubits in chain.

    Returns
    -------
    psi : 2D array
        State vector.
    """
    psi = np.array([[1], [0]])
    for i in range(N - 1):
        psi = np.kron(psi, np.array([[0], [1]]))
    return psi


def first_up(N):
    """
    Generate state for Ising chain with all spins down except the first.

    Parameters
    ----------
    N : int
        Number of qubits in chain.

    Returns
    -------
    psi : 2D array
        State vector.
    """
    psi = np.array([[0], [1]])
    for i in range(N - 1):
        psi = np.kron(psi, np.array([[1], [0]]))
    return psi


def ss_rho(psi):
    """
    Turn a single state vector into a density matrix.

    ρ = |ψ><ψ|

    Parameters
    ----------
    psi : 2D array
        State vector.

    Returns
    -------
    rho : 2D array
        Density matrix.

    """
    return np.outer(psi, psi)


def partial_trace(rho, keep, dims, optimize=False):
    """
    Calculate the partial trace.

    ρ_a = Tr_b(ρ)

    Parameters
    ----------
    rho : 2D array
        Matrix to trace.
    keep : array
        An array of indices of the spaces to keep after being traced. For
        instance, if the space is A x B x C x D and we want to trace out B
        and D, keep = [0,2].
    dims : array
        An array of the dimensions of each space. For instance, if the
        space is A x B x C x D, dims = [dim_A, dim_B, dim_C, dim_D].

    Returns
    -------
    ρ_a : 2D array
        Traced matrix.
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims, 2))
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)


def vn_entropy(rho):
    """
    Calculate von Neumann entropy for given density matrix.

    S = Tr(ρ) Log(ρ)

    Parameters
    ----------
    rho : 2D array
        Density matrix.

    Returns
    -------
    S : float
        Entropy.
    """
    log = la.logm(rho) / np.log(2)
    R = np.matmul(rho, log)
    S = - np.trace(R)
    return S


def information_subregion(rho, num_keep):
    """
    Calculate mutual information between reference and first n qubits.

    Parameters
    ----------
    rho : 2D array
        Density matrix.
    num_keep : int
        Number of qubits in information set.

    Returns
    -------
    I : float
        Mutual information.
    """
    size = int(np.round(np.log2(len(rho))))
    ref_keep = [size - 1]
    sub_keep = []
    jnt_keep = []
    dims = []
    for i in range(size):
        dims.append(2)
    for i in range(num_keep):
        sub_keep.append(i)
    for i in range(size):
        if i in ref_keep or i in sub_keep:
            jnt_keep.append(i)
    reference = partial_trace(rho, ref_keep, dims)
    subregion = partial_trace(rho, sub_keep, dims)
    joint = partial_trace(rho, jnt_keep, dims)
    ref_S = vn_entropy(reference)
    sub_S = vn_entropy(subregion)
    jnt_S = vn_entropy(joint)
    return np.real(ref_S + sub_S - jnt_S)


def information_pointwise(rho, qubits_considered):
    """
    Calculate mutual information between reference and the individual
    qubits specified.

    Parameters
    ----------
    rho : 2D array
        Density matrix.
    qubits_considered : list
        A list of ints specifying the idices of the qubits.

    Returns
    -------
    info_results : list
        A list of the mutual information of all the specified qubits in
        the same order they were provided.
    """
    size = int(np.round(np.log2(len(rho))))
    dims = []
    ref_keep = [size - 1]
    info_results = []
    for i in range(size):
        dims.append(2)
    for l in qubits_considered:
        sub_keep = [l]
        jnt_keep = []
        for i in range(size):
            if i in ref_keep or i in sub_keep:
                jnt_keep.append(i)
        reference = partial_trace(rho, ref_keep, dims)
        subregion = partial_trace(rho, sub_keep, dims)
        joint = partial_trace(rho, jnt_keep, dims)
        ref_S = vn_entropy(reference)
        sub_S = vn_entropy(subregion)
        jnt_S = vn_entropy(joint)
        info_results.append(np.real(ref_S + sub_S - jnt_S))
    return info_results


def evolve(hamiltonian, time):
    """
    Generate the time evolution operator given the hamiltonian H. This uses
    units where either ħ=1 or H includes a factor 1/ħ already.

    U = e^(-iHt/ħ)

    Parameters
    ----------
    hamiltonian : 2D array
        Hamiltonian matrix.
    time : float
        Time to evolve to.

    Returns
    -------
    U : 2D array
        Time evolution operator.
    """
    U = la.expm(- 1j * time * hamiltonian)
    return la.expm(- 1j * time * hamiltonian)


def info_helper(inputs):
    """
    Helper to compute information_subregion with multiprocessing support.

    Parameters
    ----------
    inputs : list, [2D array, int, 2D array, float]
        Index 0 is the density operator.
        Index 1 is the number n of qubits specifying the subregion.
        Index 2 is the hamiltonian.
        Index 3 is the time to evolve.

    Returns
    -------
    I : float
        mutual information

    See Also
    --------
    information_subregion :
        Mutual information between reference and first n qubits.
    """
    rho = inputs[0]
    num_keep = inputs[1]
    H = inputs[2]
    time = inputs[3]
    U = np.matrix(evolve(H, time))
    rho_t = np.array(np.matmul(U, np.matmul(rho, U.H)))
    info = information_subregion(rho_t, num_keep)
    return info


def pointwise_helper(inputs):
    """
    Helper to compute information_pointwise with multiprocessing support.

    Parameters
    ----------
    inputs : list, [2D array, list, 2D array, float]
        Index 0 is the density operator.
        Index 1 is a list of ints to find mutual information with.
        Index 2 is the hamiltonian.
        Index 3 is the time to evolve.

    Returns
    -------
    info_list : list
        List of mutual information with each specified qubit.

    See Also
    --------
    information_pointwise :
        Mutual information between reference and each individual specified
        qubit in a list.
    """
    rho = inputs[0]
    qubits_considered = inputs[1]
    H = inputs[2]
    time = inputs[3]
    U = np.matrix(evolve(H, time))
    rho_t = np.array(np.matmul(U, np.matmul(rho, U.H)))
    info = information_pointwise(rho_t, qubits_considered)
    return info


def mutual_information_over_time(total_number_qubits,
                                 subregion_number_qubits, time_array,
                                 number_cores=5):
    """
    Compute mutual information between first n qubit subregion A(n) and
    the reference qubit over time.

    Parameters
    ----------
    total_number_qubits : int
        Length of the Ising chain.
    subregion_number_qubits : int
        Number of qubits to include in subregion A(n).
    time_array : 1D array
        Times to evolve the system to.
    number_cores : int, optional
        Number of cores to use for processing.

    Returns
    -------
    info_array : 1D array
        The mutual information at each time in the time_array.
    """
    N = total_number_qubits
    l = subregion_number_qubits
    H = np.kron(chain_hamiltonian(N, 1, 0, 1), np.array([[1, 0], [0, 1]]))
    psi1 = all_up(N)
    psi2 = first_down(N)
    psi3 = first_up(N)  # Unused.
    ref_down = np.array([[1], [0]])
    ref_up = np.array([[0], [1]])
    big_bell = (np.kron(psi1, ref_down) + np.kron(psi2, ref_up))
    big_bell *= (1 / np.sqrt(2))
    rho = np.outer(big_bell, big_bell)
    inputs = []
    for i in range(len(time_array)):
        inputs.append((rho, l, H, time_array[i]))
    with Pool(number_cores) as p:
        info_list = p.map(info_helper, inputs)
    info_array = np.array(info_list)
    return info_array


def pointwise_mutual_information_over_time(total_number_qubits,
                                           qubits_considered, time_array,
                                           number_cores=5):
    """
    Compute mutual information betweeen each individual qubit in the
    specified list and the reference qubit over time.

    Parameters
    ----------
    total_number_qubits : int
        Length of the Ising chain.
    qubits_considered : list
        List of ints specifying the indices of the qubits to calculate the
        mutual information for.
    time_array : 1D array
        Times to evolve the system to.
    number_cores : int, optional
        Number of cores to use for processing.

    Returns
    -------
    info_array : 2D array
        The mutual information at each time in the time array for each
        qubit in the list of qubits considered.
    """
    N = total_number_qubits
    H = np.kron(chain_hamiltonian(N, 1, 0, 1), np.array([[1, 0], [0, 1]]))
    psi1 = all_up(N)
    psi2 = first_down(N)
    psi3 = first_up(N)  # Unused.
    ref_down = np.array([[1], [0]])
    ref_up = np.array([[0], [1]])
    big_bell = (np.kron(psi1, ref_down) + np.kron(psi2, ref_up))
    big_bell *= (1 / np.sqrt(2))
    rho = np.outer(big_bell, big_bell)
    inputs = []
    for i in range(len(time_array)):
        inputs.append((rho, qubits_considered, H, time_array[i]))
    with Pool(number_cores) as p:
        info_list = p.map(pointwise_helper, inputs)
    info_array = np.array(info_list)
    return np.transpose(info_array)
