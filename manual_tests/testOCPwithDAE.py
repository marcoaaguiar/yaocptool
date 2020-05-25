from casadi import mtimes

from yaocptool.methods import DirectMethod
from yaocptool.modelling import SystemModel, OptimalControlProblem

# create model
model = SystemModel(name='dae_system')

x = model.create_state('x', 2)
y = model.create_algebraic_variable('y', 2)
u = model.create_control('u')
a = model.create_parameter('a')
b = model.create_theta('b')

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
# problem.L = mtimes(x.T, x) + u ** 2
problem.S = mtimes(x.T, x) + u ** 2 + b ** 2
problem.x_0 = [0, 1]
problem.set_theta_as_optimization_theta(b, -.5, .5)
# problem.include_equality(problem.p_opt + 0.25)
problem.include_time_inequality(+u + x[0], when='end')

# instantiate a solution method
solution_method = DirectMethod(problem,
                               discretization_scheme='collocation',
                               degree_control=1,
                               )
# theta = create_constant_theta(1, 1, 10)

# initial_guess = solution_method.discretizer.create_initial_guess_with_simulation(p=[1])
# solution = solution_method.solve(p=[1], theta=theta, initial_guess=initial_guess)
solution = solution_method.solve(p=[1], x_0=[2, 3, 0])

solution.plot([{'x': ['x_0', 'x_1']}, {'y': ['y_0', 'y_1']}, {'x': [0, 1], 'u': ['u']}, {'theta_opt': ['b']}])

solution.to_dataset()
