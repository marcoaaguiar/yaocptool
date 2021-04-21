from casadi import mtimes

from yaocptool.methods import DirectMethod
from yaocptool.modelling import OptimalControlProblem, SystemModel


def get_model(name="dae_system"):
    model = SystemModel(name=name, model_name_as_prefix=True)

    x = model.create_state("x", 2)
    y = model.create_algebraic_variable("y", 2)
    u = model.create_control("u")
    a = model.create_parameter("a")

    # model.include_system_equations(ode=[
    #     -a * x[0] + y[0],
    #     -x[1] + y[1] + u[0]
    # ], alg=[
    #     -y[0] - x[1] ** 2,
    #     -y[1] - x[0] ** 1
    # ])
    model.include_equations(
        ode=[-a * x[0] + y[0], -x[1] + y[1] + u[0]], alg=[-y[0] - x[1], -y[1] - x[0]]
    )
    return model


def get_ocp(model):
    # create ocp
    problem = OptimalControlProblem(model)
    problem.t_f = 10
    problem.L = mtimes(model.x.T, model.x) + model.u ** 2
    problem.x_0 = [0, 1]
    problem.include_time_inequality(+model.u + model.x[0], when="end")

    return problem


# create ocp
problem = get_ocp(get_model())
problem.create_adjoint_states()
# instantiate a solution method
solution_method = DirectMethod(
    problem,
    discretization_scheme="collocation",
    degree_control=1,
    degree=4,
)

solution = solution_method.solve(p=[1])

solution.plot([{"x": [0, 1]}, {"x": [2, 3]}, {"y": [0, 1]}, {"y": [2, 3]}, {"u": [0]}])

solution.to_dataset()
