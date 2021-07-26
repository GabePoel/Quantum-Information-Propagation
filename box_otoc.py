import numpy as np
from scipy import linalg as la
from multiprocessing import Pool


def box_hamiltonian(highest_level, hbar=1, mass=1, length=1):
    """
    Hamiltonian for particle in a box.

    Parameters
    ----------
    highest_level : int
        Highest energy level to include in hamiltonian.
    hbar : float, optional
        Value to use for reduced Planck constant.
    mass : float, optional
        Mass of particle.
    length : float, optional
        Length of box.

    Returns
    -------
    H : 2D array
        Hamiltonian matrix.
    """
    constant_factor = np.pi ** 2 * hbar ** 2 / (2 * mass * length ** 2)
    matrix = np.zeros((highest_level, highest_level))
    for n in range(1, highest_level + 1):
        matrix[n - 1, n - 1] = n ** 2
    return np.matrix(constant_factor * matrix)


def partition_function(beta, hamiltonian):
    """
    Partition function for given hamiltonian under density operator
    formulation of quantum mechanics.

    Z = tr Σ e^(-βH)

    Parameters
    ----------
    beta : float
        Value of β = 1 / (k_B T) for temperature T where k_B is the
        Boltzmann constant.
    hamiltonian : 2D array
        Hamiltonian matrix.

    Returns
    -------
    Z : float
        Partition function at specified beta (temperature).
    """
    return np.trace(la.expm(-beta * hamiltonian))


def box_thermal_state(highest_level, beta, hbar=1, mass=1, length=1):
    """
    Generate thermal state for particle in a box using the density operator
    formulation of quantum mechanics.

    ρ = Σ e^(-βE_n) |n><n|

    Parameters
    ----------
    highest_level : int
        Highest energy level to include in hamiltonian.
    beta : float
        Value of β = 1 / (k_B T) for temperature T where k_B is the
        Boltzmann constant.
    hbar : float, optional
        Value to use for reduced Planck constant.
    mass : float, optional
        Mass of particle.
    length : float, optional
        Length of box.

    Returns
    -------
    ρ : 2D array
        Thermal state matrix.
    """
    H = box_hamiltonian(highest_level, hbar, mass, length)
    return np.matrix(la.expm(- beta * H) / partition_function(beta, H))


def time_evolution(hamiltonian, time, hbar=1):
    """
    Generate the time evolution operator given the hamiltonian H.

    U = e^(-iHt/ħ)

    Parameters
    ----------
    hamiltonian : 2D array
        Hamiltonian matrix.
    time : float
        Time to evolve to.
    hbar : float, optional
        Value to use for reduced Planck constant.

    Returns
    -------
    U : 2D array
        Time evolution operator.
    """
    return np.matrix(la.expm(-1j * hamiltonian * time / hbar))


def box_position_operator(highest_level, hbar, mass, length, time,
                          explicit=False):
    """
    Calculation of matrix representation of position operator for a
    particle in a box in the energy basis.

    Parameters
    ----------
    highest_level : int
        Highest energy level to include in hamiltonian.
    hbar : float
        Value to use for reduced Planck constant.
    mass : float
        Mass of particle.
    length : float
        Length of box.
    time : float
        Time to evolve to.
    explicit : bool
        Whether to calculate the matrix explicitly or use the identity I
        derived in the paper's appendix.

    Returns
    -------
    x : 2D array
        Box position operator.
    """
    matrix = np.zeros((highest_level, highest_level), dtype=np.complex128)
    for n in range(1, highest_level + 1):
        En = n ** 2 * hbar ** 2 * n ** 2 / (2 * mass * length ** 2)
        for m in range(1, highest_level + 1):
            Em = m ** 2 * hbar ** 2 * m ** 2 / (2 * mass * length ** 2)
            phase = np.e ** (-1j * (Em - En) * time / hbar)
            if m == n:
                amplitude = length / 2
            else:
                if explicit:
                    # Explicit amplitude calculation.
                    amplitude = 2 * length * \
                        (-2 * m * n + 2 * (-1) ** (m + n) * m * n) / (
                            (m ** 2 - n ** 2) ** 2 * np.pi ** 2)
                else:
                    # Amplitude calculation from appendix.
                    amplitude = (length / np.pi ** 2) * (1 - (-1) ** (n + m)
                        ) * (1 / ((n + m) ** 2) - 1 / ((n - m) ** 2))
            matrix[n - 1, m - 1] = phase * amplitude
    return np.matrix(matrix)


def box_momentum_operator(highest_level, hbar, length):
    """
    DEPRECATED.

    Calculation of the matrix representation of the momentum operator for
    a particle in a box in the energy basis. Assumes a mass of m = 1.

    Parameters
    ----------
    highest_level : int
        Highest energy level to include in hamiltonian.
    hbar : float
        Value to use for reduced Planck constant.
    length : float
        Length of box.

    Returns
    -------
    p : 2D array
        Box momentum operator.

    See Also
    --------
    momentum_operator : Preferred way to compute this.
    """
    matrix = np.zeros((highest_level, highest_level), dtype=np.complex128)
    for n in range(1, highest_level + 1):
        kn = n * np.pi / length
        matrix[n - 1, n - 1] = hbar * kn
    return np.matrix(matrix)


def momentum_operator(position_operator, hamiltonian):
    """
    Calculation of the matrix representation of the momentum operator for
    arbitrary single particle system where the potential commutes with the
    position operator. Done in energy eigenbasis.

    Parameters
    ----------
    position_operator : 2D array
        Matrix representation of position operator in energy basis.
    hamiltonian : 2D array
        Matrix representation of the hamiltonian in energy basis.

    Returns
    -------
    p : 2D array
        Matrix representation of momentum operator in energy basis.
    """
    matrix = np.zeros(position_operator.shape, dtype=np.complex128)
    highest_level = position_operator.shape[0]
    for n in range(1, highest_level + 1):
        En = hamiltonian[n - 1, n - 1]
        for m in range(1, highest_level + 1):
            Em = hamiltonian[m - 1, m - 1]
            elmnt = (1j / 2) * (En - Em) * position_operator[n - 1, m - 1]
            matrix[n - 1, m - 1] = elmnt
    return np.matrix(matrix)


def energy_expectation(highest_level, hbar, mass, length, beta):
    """
    Computes the expected energy of a particle in a box in a thermal state
    under the density operator formulation of quantum mechanics.

    Parameters
    ----------
    highest_level : int
        Highest energy level to include in hamiltonian.
    hbar : float
        Value to use for reduced Planck constant.
    mass : float
        Mass of particle.
    length : float
        Length of box.
    beta : float
        Value of β = 1 / (k_B T) for temperature T where k_B is the
        Boltzmann constant.

    Returns
    -------
    <H> : float
        Energy expectation.
    """
    H = box_hamiltonian(highest_level, hbar, mass, length)
    rho = box_thermal_state(highest_level, beta, hbar, mass, length)
    return np.trace(H * rho)


def energy_component(index_1, index_2):
    """
    Energy difference δE_nm as defined in paper.

    Parameters
    ----------
    index_1 : int
        An energy level n.
    index_2 : int
        An energy level m.

    Returns
    -------
    δE_nm : float
        Energy difference.
    """
    return np.pi ** 2 * (index_1 ** 2 - index_2 ** 2)


def position_component(index_1, index_2):
    """
    Specific component of matrix representation of the position operator
    for a particle in a box in the energy eigenbasis.

    Parameters
    ----------
    index_1 : int
        An energy level n.
    index_2 : int
        An energy level m.

    Returns
    -------
    x_nm : float
        Position element.
    """
    if index_1 == index_2:
        return 1 / 2
    else:
        return (((1 - (-1) ** (index_1 + index_2)) / np.pi ** 2) *
                (1 / (index_1 + index_2) ** 2 - 1 / (index_1 - index_2) ** 2))


def momentum_component(index_1, index_2):
    """
    Specific component of matrix representation of the momentum operator
    for a particle in a box in the energy eigenbasis. Method derived in
    paper's appendix.

    Parameters
    ----------
    index_1 : int
        An energy level n.
    index_2 : int
        An energy level m.

    Returns
    -------
    p_nm : momentum
        Momentum element.
    """
    return (1j / 2) * (energy_component(index_1, index_2) *
                       position_component(index_1, index_2))


def paper_partition_function(highest_level, temperature):
    """
    Partition function for given particle in a box under density operator
    formulation of quantum mechanics.

    Z = tr Σ e^(-βH)

    Parameters
    ----------
    highest_level : int
        Highest energy level to include in hamiltonian.
    temperature : float
        Environment temperature of system. T = 1 / (β k_B) where k_B is
        the Boltzmann constant. Assumes k_B = 1.

    Returns
    -------
    Z : float
        Partition function at specified beta (temperature).
    """
    z_partial = []
    for i in range(highest_level):
        E = np.pi ** 2 * i ** 2
        z_partial.append(np.e ** (-E / temperature))
    return np.sum(z_partial)


def paper_box_OTOC(inputs):
    """
    Helper to compute out-of-time-ordered-correlator for particle in a box
    using the method I specified in the paper with multiprocessing support.

    Parameters
    ----------
    inputs : list, [int, float, float]
        Index 0 is the highest energy level to consider.
        Index 1 is the temperature to evaluate OTOC at.
        Index 2 is the time to evolve the system to.

    Returns
    -------
    OTOC : float
        The out-of-time-ordered-correlator.
    """
    highest_level = inputs[0]
    temperature = inputs[1]
    time = inputs[2]
    energy_OTOC_n_list = []
    for n in range(highest_level):
        c_nm_list = []
        for m in range(highest_level):
            b_nmk_list = []
            for k in range(highest_level):
                E_nk = energy_component(n + 1, k + 1)
                E_km = energy_component(k + 1, m + 1)
                phase_left = np.e ** (1j * E_nk * time)
                phase_right = np.e ** (1j * E_km * time)
                x_nk = position_component(n + 1, k + 1)
                p_km = momentum_component(k + 1, m + 1)
                p_nk = momentum_component(n + 1, k + 1)
                x_km = position_component(k + 1, m + 1)
                b_nmk = -1j * (phase_left * x_nk * p_km - phase_right *
                               p_nk * x_km)
                b_nmk_list.append(b_nmk)
            b_nmk_array = np.array(b_nmk_list)
            b_nm = np.sum(b_nmk_array)
            c_nm = b_nm * np.conj(b_nm)
            c_nm_list.append(c_nm)
        c_nm_array = np.array(c_nm_list)
        c_n = np.sum(c_nm_array)
        energy_OTOC_n = np.e ** (- np.pi ** 2 * n ** 2 / temperature) * c_n
        energy_OTOC_n_list.append(energy_OTOC_n)
    energy_OTOC_n_array = np.array(energy_OTOC_n_list)
    OTOC = (np.sum(energy_OTOC_n_array) /
            paper_partition_function(highest_level, temperature))
    return OTOC


def postition_position_OTOC(inputs):
    """
    Helper to compute out-of-time-ordered-correlator for particle in a box
    using the method I specified in the paper with multiprocessing support.
    This is the OTOC for the position with itself evolved in time. It's not
    used in the paper but is proportional to the momentum result used.

    Parameters
    ----------
    inputs : list, [int, float, float]
        Index 0 is the highest energy level to consider.
        Index 1 is the temperature to evaluate OTOC at.
        Index 2 is the time to evolve the system to.

    Returns
    -------
    OTOC : float
        The out-of-time-ordered-correlator.
    """
    highest_level = inputs[0]
    temperature = inputs[1]
    time = inputs[2]
    energy_OTOC_n_list = []
    for n in range(highest_level):
        c_nm_list = []
        for m in range(highest_level):
            b_nmk_list = []
            for k in range(highest_level):
                E_nk = energy_component(n + 1, k + 1)
                E_km = energy_component(k + 1, m + 1)
                phase_left = np.e ** (1j * E_nk * time)
                phase_right = np.e ** (1j * E_km * time)
                x_nk = position_component(n + 1, k + 1)
                p_km = position_component(k + 1, m + 1)
                p_nk = position_component(n + 1, k + 1)
                x_km = position_component(k + 1, m + 1)
                b_nmk = -1j * (phase_left * x_nk * p_km - phase_right *
                               p_nk * x_km)
                b_nmk_list.append(b_nmk)
            b_nmk_array = np.array(b_nmk_list)
            b_nm = np.sum(b_nmk_array)
            c_nm = b_nm * np.conj(b_nm)
            c_nm_list.append(c_nm)
        c_nm_array = np.array(c_nm_list)
        c_n = np.sum(c_nm_array)
        energy_OTOC_n = np.e ** (- np.pi ** 2 * n ** 2 / temperature) * c_n
        energy_OTOC_n_list.append(energy_OTOC_n)
    energy_OTOC_n_array = np.array(energy_OTOC_n_list)
    OTOC = (np.sum(energy_OTOC_n_array) /
            paper_partition_function(highest_level, temperature))
    return OTOC


def paper_box_OTOC_over_time(highest_level, temperature_array, time_array,
                             number_cores=5):
    """
    Compute out-of-time-ordered-correlator for particle in a box using the
    method I specified in the paper with multiprocessing support over time.

    Parameters
    ----------
    highest_level : int
        Highest energy level to include in hamiltonian.
    temperature_array : 1D array
        Temperatures of the system at each specified time.
    time_array : 1D array
        Times to evolve the system to.
    number_cores : int, optional
        Number of cores to use for processing.

    Returns
    -------
    OTOC_arry : 1D array
        The out-of-time-ordered-correlator at each specified time.
    """
    inputs = []
    for i in range(len(temperature_array)):
        inputs.append((highest_level, temperature_array[i], time_array[i]))
    with Pool(number_cores) as p:
        OTOC_list = p.map(paper_box_OTOC, inputs)
    OTOC_array = np.array(OTOC_list)
    return OTOC_array


def position_position_OTOC_over_time(highest_level, temperature_array,
                                     time_array, number_cores=5):
    """
    Compute out-of-time-ordered-correlator for particle in a box using the
    method I specified in the paper with multiprocessing support over time.
    This is the OTOC for the position with itself evolved in time. It's not
    used in the paper but is proportional to the momentum result used.

    Parameters
    ----------
    highest_level : int
        Highest energy level to include in hamiltonian.
    temperature_array : 1D array
        Temperatures of the system at each specified time.
    time_array : 1D array
        Times to evolve the system to.
    number_cores : int, optional
        Number of cores to use for processing.

    Returns
    -------
    OTOC_arry : 1D array
        The out-of-time-ordered-correlator at each specified time.
    """
    inputs = []
    for i in range(len(temperature_array)):
        inputs.append((highest_level, temperature_array[i], time_array[i]))
    with Pool(number_cores) as p:
        OTOC_list = p.map(paper_box_OTOC, inputs)
    OTOC_array = np.array(OTOC_list)
    return OTOC_array


def box_OTOC_beta(highest_level, hbar, mass, length, beta, time):
    """
    OTOC for particle in a box using beta (inverse of temperature).

    Parameters
    ----------
    highest_level : int
        Highest energy level to include in hamiltonian.
    hbar : float
        Value to use for reduced Planck constant.
    mass : float
        Mass of particle.
    length : float
        Length of box.
    beta : float
        Value of β = 1 / (k_B T) for temperature T where k_B is the
        Boltzmann constant.
    time : float
        Time to evolve to.

    Returns
    -------
    OTOC : complex
        Expectation of OTOC.
    """
    rho = box_thermal_state(highest_level, beta, hbar, mass, length)
    x = box_position_operator(highest_level, hbar, mass, length, time)
    p = box_momentum_operator(highest_level, hbar, length)
    commutator = x * p - p * x
    otoc_operator = commutator ** 2
    expectation = np.trace(rho * otoc_operator)
    return - expectation


def box_OTOC_temp(highest_level, hbar, mass, length, boltzmann, temp, time,
                  explicit=False):
    """
    OTOC for particle in a box using temperature.

    Parameters
    ----------
    highest_level : int
        Highest energy level to include in hamiltonian.
    hbar : float
        Value to use for reduced Planck constant.
    mass : float
        Mass of particle.
    length : float
        Length of box.
    temp : float
        Environment temperature of system. T = 1 / (β k_B) where k_B is
        the Boltzmann constant. Assumes k_B = 1.
    time : float
        Time to evolve to.
    explicit : bool
        Whether to calculate the matrix explicitly or use the identity I
        derived in the paper's appendix.

    Returns
    -------
    OTOC : complex
        Expectation of OTOC.
    """
    beta = 1 / (boltzmann * temp)
    H = box_hamiltonian(highest_level, hbar, mass, length)
    rho = box_thermal_state(highest_level, beta, hbar, mass, length)
    x = box_position_operator(highest_level, hbar, mass, length, time,
                              explicit=explicit)
    x0 = box_position_operator(highest_level, hbar, mass, length, 0,
                               explicit=explicit)
    p = momentum_operator(x0, H)
    commutator = x * p - p * x
    otoc_operator = commutator * commutator.H
    values = otoc_operator * rho
    expectation = np.trace(values)
    return expectation


def box_OTOC_temp_over_time(highest_level, hbar, mass, length, boltzmann,
                            temp, min_time, max_time, explicit=False):
    """
    OTOC for particle in a box over all specified temperatures.

    Parameters
    ----------
    highest_level : int
        Highest energy level to include in hamiltonian.
    hbar : float
        Value to use for reduced Planck constant.
    mass : float
        Mass of particle.
    length : float
        Length of box.
    boltzmann : float
        Boltzmann constant.
    temp : float
        Environment temperature of system. T = 1 / (β k_B) where k_B is
        the Boltzmann constant.
    min_time : float
        Time to start at.
    max_time : float
        Time to end at.
    explicit : bool
        Whether to calculate the matrix explicitly or use the identity I
        derived in the paper's appendix.

    Returns
    -------
    OTOC_array : 1D array
        Array of expectation of OTOCs for all temperatures.
    """
    delta_t = 1 / 1000
    t = np.linspace(min_time, max_time, np.int(np.round((max_time -
                                                         min_time) / delta_t)))
    c = np.zeros(t.shape)
    for i in range(len(c)):
        c[i] = box_OTOC_temp(highest_level, hbar, mass, length, boltzmann,
                             temp, delta_t * i + min_time, explicit=explicit)
    return t, c
