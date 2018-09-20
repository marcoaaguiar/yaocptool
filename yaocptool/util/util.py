from casadi import is_equal, DM, vec, vertcat, substitute, mtimes, integrator, MX, repmat


def find_variables_indices_in_vector(var, vector):
    index = []
    for j in range(vector.size1()):
        for i in range(var.numel()):
            if is_equal(vector[j], var[i]):
                index.append(j)
    return index


def remove_variables_from_vector(var, vector):
    """
        Returns a vector with items removed
    :param var: items to be removed
    :param vector: vector which will have items removed
    :return:
    """
    vector = vector[:]
    to_remove = find_variables_indices_in_vector(var, vector)
    to_remove.sort(reverse=True)
    for it in to_remove:
        vector.remove([it], [])
    return vector


def remove_variables_from_vector_by_indices(vector, indices):
    """
        Returns a vector with items removed
    :param vector: vector which will have items removed
    :param list indices: list of indices for which the variables need to be removed.
    :return:
    """
    vector = vector[:]
    indices.sort(reverse=True)
    for it in indices:
        vector.remove([it], [])
    return vector


def create_constant_theta(constant, dimension, finite_elements):
    theta = {}
    for i in range(finite_elements):
        theta[i] = vec(constant * DM.ones(dimension, 1))

    return theta


def join_thetas(*args):
    new_theta = {}
    all_keys = []
    n_keys = 0
    for theta in args:
        theta_keys = theta.keys()
        n_keys = len(theta_keys) if n_keys == 0 else n_keys
        if len(theta_keys) > 0 and not len(theta_keys) == n_keys:
            raise ValueError('Only thetas with size zero or same size are accepted, given: {}'.format(
                [len(th.keys()) for th in args]))

        all_keys.extend(theta.keys())
    all_keys = list(set(all_keys))

    for i in all_keys:
        new_theta[i] = []
        for theta in args:
            if i in theta:
                theta1_value = theta[i]
            else:
                theta1_value = []

            new_theta[i] = vertcat(new_theta[i], theta1_value)

    return new_theta


def convert_expr_from_tau_to_time(expr, t_sym, tau_sym, t_k, t_kp1):
    h = t_kp1 - t_k
    return substitute(expr, tau_sym, (t_sym - t_k) / h)


def blockdiag(matrices_list):
    """Receives a list of matrices and return a block diagonal.

    :param list matrices_list: list of matrices
    """

    size_1 = sum([m.size1() for m in matrices_list])
    size_2 = sum([m.size2() for m in matrices_list])

    matrix = DM.zeros(size_1, size_2)
    index_1 = 0
    index_2 = 0

    for m in matrices_list:
        matrix[index_1: index_1 + m.size1(), index_2: index_2 + m.size2()] = m
        index_1 += m.size1()
        index_2 += m.size2()

    return matrix


def expm(a_matrix):
    """Since casadi does not have native support for matrix exponential, this is a trick to computing it.
    It can be quite expensive, specially for large matrices.
    THIS ONLY SUPPORT NUMERIC MATRICES, DOES NOT SUPPORT SX SYMBOLIC VARIABLES.

    :param DM a_matrix: matrix
    :return:
    """
    dim = a_matrix.shape[1]

    # Create the integrator
    x = MX.sym('x', a_matrix.shape[1])
    a_mx = MX.sym('x', a_matrix.shape)
    ode = mtimes(a_mx, x)
    dae_system_dict = {'x': x, 'ode': ode, 'p': vec(a_mx)}

    integrator_ = integrator("integrator", "cvodes", dae_system_dict, {'tf': 1})
    integrator_MAP = integrator_.map(a_matrix.shape[1], 'thread')

    res = integrator_MAP(x0=DM.eye(dim), p=repmat(vec(a_matrix), (1, a_matrix.shape[1])))['xf']

    return res
