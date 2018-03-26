from casadi import is_equal, DM, vec, vertcat, substitute


def find_variables_indices_in_vector(var, vector):
    index = []
    for j in range(vector.size1()):
        for i in range(var.numel()):
            if is_equal(vector[j], var[i]):
                index.append(j)
    return index


def remove_variables_from_vector(var, vector):
    to_remove = find_variables_indices_in_vector(var, vector)
    to_remove.sort(reverse=True)
    for it in to_remove:
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
