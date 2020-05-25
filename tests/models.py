from yaocptool.modelling import OptimalControlProblem, SystemModel
from casadi import DM, mtimes


def _create_linear_system(n_x, n_u, a, b, name):
    model = SystemModel(name=name, n_x=n_x, n_u=n_u)
    x = model.x_sym
    u = model.u_sym
    model.include_system_equations(mtimes(a, x) + mtimes(b, u))
    return model


def create_siso():
    a = DM([-1])
    b = DM([1])

    model = _create_linear_system(n_x=1, n_u=1, a=a, b=b, name='SISO')
    problem = OptimalControlProblem(model, obj={'Q': DM.eye(1), 'R': DM.eye(1)}, x_0=[1])
    return model, problem


def create_2x1_mimo():
    a = DM([[-1, -2], [5, -1]])
    b = DM([1, 0])

    model = _create_linear_system(n_x=2, n_u=1, a=a, b=b, name='MIMO_2x1')
    problem = OptimalControlProblem(model, obj={'Q': DM.eye(2), 'R': DM.eye(1)}, x_0=[1, 1])
    return model, problem


def create_2x2_mimo():
    a = DM([[-1, -2], [5, -1]])
    b = DM([[1, 0], [0, 1]])

    model = _create_linear_system(n_x=2, n_u=2, a=a, b=b, name='MIMO_2x2')
    problem = OptimalControlProblem(model, obj={'Q': DM.eye(2), 'R': DM.eye(2)}, x_0=[1, 1])
    return model, problem
