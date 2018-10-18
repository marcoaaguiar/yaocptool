from os.path import dirname, abspath
from sys import path

path.append(abspath(dirname(dirname(__file__))))

from yaocptool.methods import DirectMethod
from yaocptool.modelling import SystemModel, OptimalControlProblem

model = SystemModel(n_x=1, n_u=1)
model.include_system_equations(ode=(-model.x_sym[0] + model.u_sym[0]))

problem = OptimalControlProblem(model, obj={'Q': 1, 'R': 1}, x_0=[1])
problem.u_min[0] = 0
problem.u_max[0] = 1

problem.x_min[0] = 0.6

solution_method = DirectMethod(problem, degree_control=3,
                               discretization_scheme='collocation',
                               degree=3, finite_elements=5, integrator_type='implicit')

solution = solution_method.solve()

solution.plot([{'x': [0, 1]}, {'u': [0]}])
