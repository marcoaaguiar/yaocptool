from casadi import mtimes

from yaocptool.methods import DirectMethod
from yaocptool.modelling import SystemModel, OptimalControlProblem

# create model
model = SystemModel(name='dae_system')

x = model.create_state('x', 2)
y = model.create_algebraic_variable('y', 2)
u = model.create_control('u')
a = model.create_parameter('a')
b = model.create_parameter('b')

model.include_system_equations(ode=[
    -a * x[0] + b * y[0],
    -x[1] + y[1] + u[0]
], alg=[
    -y[0] - x[1] ** 2,
    -y[1] - x[0] ** 1
])

# create ocp
problem = OptimalControlProblem(model)
problem.t_f = 10
problem.L = mtimes(x.T, x) + u ** 2
problem.x_0 = [0, 1]
problem.set_parameter_as_optimization_parameter(b, -.5, .5)
# problem.include_equality(problem.p_opt + 0.25)
problem.include_time_inequality(-u - 0.05)

# instantiate a solution method
solution_method = DirectMethod(problem,
                               discretization_scheme='collocation',
                               degree_control=3,
                               )
solution = solution_method.solve(p=[1, 2])

solution.plot([{'x': [0, 1]}, {'y': [0, 1]}, {'u': [0]}])
