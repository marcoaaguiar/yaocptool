import re
from typing import List, Optional, Union, Tuple, Dict, TYPE_CHECKING

from casadi import (
    DM,
    is_equal,
    vec,
    vertcat,
    substitute,
    mtimes,
    integrator,
    MX,
    repmat,
    OP_LT,
    OP_LE,
    OP_EQ,
    SX,
    collocation_points,
)

if TYPE_CHECKING:
    from casadi import FunctionCallArgT, NumericMT, SymbolicMT
else:
    FunctionCallArgT = "FunctionCallArgT"
    NumericMT = "NumericMT"
    SymbolicMT = "SymbolicMT"


def find_variables_indices_in_vector(
    var: SymbolicMT, vector: SymbolicMT, depth=0
) -> List[int]:
    """
        Given symbolic variables return the indices of the variables in a vector

    :param list|casadi.SX|casadi.MX var:
    :param casadi.SX|casadi.MX vector:
    :param int depth: depth for which is_equal will check for equality
    :return: list of indices
    :rtype: list
    """
    indices: List[int] = []
    for v in var.nz:
        for ind_vector, vector_var in enumerate(vector.nz):
            if is_equal(vector_var, v, depth):
                indices.append(ind_vector)

    return indices


def find_variables_in_vector_by_name(
    names: Union[str, List[str]], vector: Union[SX, MX], exact: bool = False
):
    """

    :param str|list of str names: variable names
    :param casadi.SX|casadi.MX vector:
    :param bool exact: default: False. If true it will use an exact match otherwise it will use an regex match
    :return: list of variables found.
    :rtype: list
    """
    if not isinstance(names, list):
        names = [names]

    result = []
    if exact:
        for name in names:
            for i in range(vector.numel()):
                if vector[i].name() == name:
                    result.append(vector[i])
    else:
        for regex in names[:]:
            result.extend(
                [
                    vector[i]
                    for i in range(vector.numel())
                    if re.match(regex, vector[i].name())
                ]
            )

    return result


def remove_variables_from_vector(var: SymbolicMT, vector: SymbolicMT) -> SymbolicMT:
    """
        Returns a vector with items removed

    :param var: items to be removed
    :param vector: vector which will have items removed
    :return:
    """
    to_remove = find_variables_indices_in_vector(var, vector)
    if len(to_remove) == 0:
        raise ValueError(
            '"var" not found in "vector", var={}, vector={}'.format(var, vector)
        )
    return remove_variables_from_vector_by_indices(to_remove, vector)


def remove_variables_from_vector_by_indices(
    indices: List[int], vector: NumericMT
) -> NumericMT:
    """
        Returns a vector with items removed

    :param list indices: list of indices for which the variables need to be removed.
    :param SX|MX|DM vector: vector which will have items removed
    :return: the vector with the variables removed
    :rtype: SX|MX|DM
    """
    if indices:
        numel = vector.numel()
        if max(indices) > numel:
            raise ValueError(
                "Found indices out of bounds. Index violating: {}, indices: {}".format(
                    max(indices), indices
                )
            )
        if min(indices) < -numel:
            raise ValueError(
                "Found indices out of bounds. Index violating: {}, indices: {}".format(
                    min(indices), indices
                )
            )

    remaining_ind = [ind for ind in range(vector.numel()) if ind not in indices]
    vector = vector[remaining_ind]

    if vector.shape == (1, 0):
        vector = vector.reshape((0, 1))
    return vector


def create_constant_theta(
    constant: Union[float, DM],
    dimension: Union[int, Tuple[int, int]],
    finite_elements: int,
) -> Dict[int, DM]:
    """
        Create constant theta

    The created theta will be a dictionary with keys = range(finite_element) and each value will be a vector with value
    'constant' and number of rows 'dimension'

    :param constant: value of each theta entry .
    :param dimension: number of rows of the vector of each theta entry.
    :param finite_elements: number of theta entries.
    :return: constant theta
    :rtype: dict
    """
    if isinstance(dimension, int):
        dimension = (dimension, 1)

    return {i: vec(constant * DM.ones(*dimension)) for i in range(finite_elements)}


def join_thetas(*args: Optional[Dict[int, DM]]) -> Dict[int, DM]:
    """
        Join the passed 'thetas'

    Receives a list of dicts. The dicts are required to have the same keys, with the exception of empty dicts or None,
    those will be skipped. For each key, the values will be concatenated and put in the return dictionary.

    :param dict|None args: thetas to be joined
    :return: return a single dictionary containing the information of the input dict
    :rtype: dict
    """
    new_theta: Dict[int, DM] = {}
    all_keys: List[int] = []
    n_keys = 0
    for theta in args:
        if theta is not None:
            theta_keys = theta.keys()
            n_keys = len(theta_keys) if n_keys == 0 else n_keys
            if len(theta_keys) > 0 and len(theta_keys) != n_keys:
                raise ValueError(
                    "Only thetas with size zero or same size are accepted, given: {}".format(
                        [len(th.keys()) if th is not None else None for th in args]
                    )
                )
            all_keys.extend(theta.keys())

    all_keys = list(set(all_keys))

    for i in all_keys:
        new_theta[i] = vertcat(
            *[theta[i] for theta in args if (theta is not None and i in theta)]
        )

    return new_theta


def convert_expr_from_tau_to_time(
    expr: SX, t_sym: SX, tau_sym: SX, t_k: float, t_kp1: float
):
    """

    :param expr:
    :param t_sym:
    :param tau_sym:
    :param t_k:
    :param t_kp1:
    :return:
    """
    d_t = t_kp1 - t_k
    return substitute(expr, tau_sym, (t_sym - t_k) / d_t)


def expm(a_matrix: FunctionCallArgT):
    """Since casadi does not have native support for matrix exponential, this is a trick to computing it.
    It can be quite expensive, specially for large matrices.
    THIS ONLY SUPPORT NUMERIC MATRICES AND MX VARIABLES, DOES NOT SUPPORT SX SYMBOLIC VARIABLES.

    :param DM a_matrix: matrix
    :return:
    """
    dim = a_matrix.shape[1]

    # Create the integrator
    x_mx = MX.sym("x", a_matrix.shape[1])
    a_mx = MX.sym("x", a_matrix.shape)
    ode = mtimes(a_mx, x_mx)
    dae_system_dict = {"x": x_mx, "ode": ode, "p": vec(a_mx)}

    integrator_ = integrator("integrator", "cvodes", dae_system_dict, {"tf": 1})
    integrator_map = integrator_.map(a_matrix.shape[1], "thread")

    x0 = DM.eye(dim)
    repmat_result = repmat(vec(a_matrix), (1, a_matrix.shape[1]))

    result = integrator_map.call({"x0": x0, "p": repmat_result})
    return result["xf"]


def is_inequality(expr: Union[SX, MX]):
    """
        Return true if an expression is an inequality (e.g.: x < 1, x <= 2)

    Only supports MX expr.

    :param MX expr: symbolic expression
    :return: True if 'expr' is an inequality, False otherwise
    :rtype: bool
    """
    return isinstance(expr, (MX, SX)) and (expr.op() == OP_LE or expr.op == OP_LT)


def is_equality(expr: Union[SX, MX]):
    """Return true if an expression is an equality (e.g.: x == 2)

    Only supports MX expr.

    :param MX expr: symbolic expression
    :return: True if 'expr' is an inequality, False otherwise
    :rtype: bool
    """
    return isinstance(expr, (MX, SX)) and expr.op() == OP_EQ


def _create_lagrangian_polynomial_basis(
    tau: SX, degree: int, point_at_zero: bool = False
) -> List[SX]:
    tau_root = collocation_points(
        degree, "radau"
    )  # type: list # All collocation time points

    if point_at_zero:
        tau_root.insert(0, 0.0)

    # For all collocation points: eq 10.4 or 10.17 in Biegler's book
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    l_list = []
    for j in range(len(tau_root)):
        ell = SX(1)
        for k in range(len(tau_root)):
            if k != j:
                ell *= (tau - tau_root[k]) / (tau_root[j] - tau_root[k])
        l_list.append(ell)

    return l_list


def create_polynomial_approximation(
    tau: SX,
    size: int,
    degree: int,
    name: Union[str, List[str]] = "var_appr",
    point_at_zero: bool = False,
) -> Tuple[SX, SX]:
    """
        Create a polynomial function.

    :param casadi.SX tau:
    :param int size: size of the approximated variable (number of rows)
    :param int degree: approximation degree
    :param list|str name: name for created parameters
    :param bool point_at_zero: if the polynomial has an collocation point at tau=0
    :return: (pol, par), returns the polynomial and a vector of parameters
    :rtype: tuple
    """
    if not isinstance(name, list):
        name = [name + "_" + str(i) for i in range(size)]

    # define the number of parameters
    n_par = degree + 1 if point_at_zero and degree > 1 else degree
    # if size = an empty symbolic variable (shape (0, n_par) is created
    if size > 0:
        points = vertcat(*[SX.sym(name[s], 1, n_par) for s in range(size)])
    else:
        points = SX.sym("empty_sx", size, n_par)

    # if the degree=1 the points are already the approximation, otherwise create a lagrangian polynomial basis
    if degree == 1:
        pol = points
    else:
        ell_list = _create_lagrangian_polynomial_basis(
            tau=tau, degree=degree, point_at_zero=point_at_zero
        )
        pol = SX(sum(ell_list[j] * points[:, j] for j in range(n_par)))
    par = vec(points)

    return pol, par
