from yaocptool.modelling import OptimalControlProblem, SystemModel
from casadi import DM, mtimes


def _create_linear_system(n_x, n_u, a, b, name):
    model = SystemModel(name=name)
    x = model.create_state("x", n_x)
    u = model.create_control("u", n_u)
    model.include_equations(ode=mtimes(a, x) + mtimes(b, u))
    return model


def create_siso():
    a = DM([-1.0])
    b = DM([1.0])

    model = _create_linear_system(n_x=1, n_u=1, a=a, b=b, name="SISO")
    problem = OptimalControlProblem(
        model, obj={"Q": DM.eye(1), "R": DM.eye(1)}, x_0=[1.0]
    )
    return model, problem


def create_2x1_mimo():
    a = DM([[-1.0, -2.0], [5.0, -1.0]])
    b = DM([1.0, 0.0])

    model = _create_linear_system(n_x=2, n_u=1, a=a, b=b, name="MIMO_2x1")
    problem = OptimalControlProblem(
        model, obj={"Q": DM.eye(2), "R": DM.eye(1)}, x_0=[1.0, 1.0]
    )
    return model, problem


def create_2x2_mimo():
    a = DM([[-1.0, -2.0], [5.0, -1.0]])
    b = DM([[1.0, 0.0], [0.0, 1.0]])

    model = _create_linear_system(n_x=2, n_u=2, a=a, b=b, name="MIMO_2x2")
    problem = OptimalControlProblem(
        model, obj={"Q": DM.eye(2), "R": DM.eye(2)}, x_0=[1.0, 1.0]
    )
    return model, problem
