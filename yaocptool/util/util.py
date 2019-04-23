import re

from casadi import is_equal, DM, vec, vertcat, substitute, mtimes, integrator, MX, repmat, OP_LT, OP_LE, OP_EQ, SX, \
    collocation_points


def find_variables_indices_in_vector(var, vector, depth=0):
    """
        Given symbolic variables return the indices of the variables in a vector

    :param casadi.SX|casadi.MX var:
    :param casadi.SX|casadi.MX vector:
    :param int depth: depth for which is_equal will check for equality
    :return: list of indices
    :rtype: list
    """
    indices = []
    for j in range(vector.size1()):
        for i in range(var.numel()):
            if is_equal(vector[j], var[i], depth):
                indices.append(j)
    return indices


def find_variables_in_vector_by_name(names, vector, exact=False):
    """

    :param str|list of str names: variable names
    :param casadi.SX|casadi.MX vector:
    :param bool exact: defautl: False. If true it will use an exact match otherwise it will use an regex match
    """
    if not isinstance(names, list):
        names = [names]

    result = []
    if exact:
        for n in names:
            for i in range(vector.numel()):
                if vector[i].name() == n:
                    result.append(vector[i])
    else:
        for regex in names[:]:
            result.extend([vector[i] for i in range(vector.numel()) if re.match(regex, vector[i].name())])

    return result


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
    THIS ONLY SUPPORT NUMERIC MATRICES AND MX VARIABLES, DOES NOT SUPPORT SX SYMBOLIC VARIABLES.

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
    integrator_map = integrator_.map(a_matrix.shape[1], 'thread')

    res = integrator_map(x0=DM.eye(dim), p=repmat(vec(a_matrix), (1, a_matrix.shape[1])))['xf']

    return res


def is_inequality(expr):
    """ Return true if an expression is an inequality (e.g.: x < 1, x <= 2)

    :param MX|SX expr: symbolic expression
    :return: True if 'expr' is an inequality, False otherwise
    :rtype: bool
    """
    if isinstance(expr, (MX, SX)):
        if expr.op() == OP_LE or expr.op == OP_LT:
            return True
    return False


def is_equality(expr):
    """ Return true if an expression is an equality (e.g.: x == 2)

    :param MX|SX expr: symbolic expression
    :return: True if 'expr' is an inequality, False otherwise
    :rtype: bool
    """
    if isinstance(expr, (MX, SX)):
        if expr.op() == OP_EQ:
            return True
    return False


def _create_lagrangian_polynomial_basis(tau, degree, starting_index=0):
    tau_root = [0] + collocation_points(degree, 'radau')  # All collocation time points

    # For all collocation points: eq 10.4 or 10.17 in Biegler's book
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    l_list = []
    for j in range(starting_index, degree + 1):
        ell = 1
        for j2 in range(starting_index, degree + 1):
            if j2 != j:
                ell *= (tau - tau_root[j2]) / (tau_root[j] - tau_root[j2])
        l_list.append(ell)

    return tau, l_list


def create_polynomial_approximation(tau, size, degree, name='var_appr', point_at_zero=False):
    """

    :param casadi.SX tau:
    :param int size: size of the approximated variable (number of rows)
    :param int degree: approximation degree
    :param list|str name: name for created parameters
    :param bool point_at_zero: if the polynomial has an collocation point at tau=0

    :return: (pol, par), returns the polynomial and a vector of parameters
    """
    if not isinstance(name, list):
        name = [name + '_' + str(i) for i in range(size)]

    if degree == 1:
        if size > 0:
            points = vertcat(*[SX.sym(name[s], 1, degree) for s in range(size)])
        else:
            points = SX.sym('empty_sx', size, degree)
        par = vec(points)
        u_pol = points

    else:
        if point_at_zero:
            if size > 0:
                points = vertcat(*[SX.sym(name[s], 1, degree + 1) for s in range(size)])
            else:
                points = SX.sym('empty_sx', size, degree)
            tau, ell_list = _create_lagrangian_polynomial_basis(tau=tau, degree=degree, starting_index=0)
            u_pol = sum([ell_list[j] * points[:, j] for j in range(0, degree + 1)])
        else:
            if size > 0:
                points = vertcat(*[SX.sym(name[s], 1, degree) for s in range(size)])
            else:
                points = SX.sym('empty_sx', size, degree)
            tau, ell_list = _create_lagrangian_polynomial_basis(tau=tau, degree=degree, starting_index=1, )
            u_pol = sum([ell_list[j] * points[:, j] for j in range(0, degree)])
        par = vec(points)

    return u_pol, par
