from sys import path
from os.path import dirname, abspath

path.append(abspath(dirname(dirname(__file__))))

from yaocptool.methods import IndirectMethod, DirectMethod, AugmentedLagrangian
from yaocptool.modelling.model_classes import SystemModel
from yaocptool.modelling.ocp import OptimalControlProblem

import time

model =  SystemModel(Nx = 1, Nu= 1)
model.include_system_equations(ode = (-model.x_sym[0] + model.u_sym[0]))

problem = OptimalControlProblem(model, obj = {'Q':1, 'R':1}, x_0 = [1])
problem.u_min[0] = 0
problem.u_max[0] = 1

problem.x_min[0] = 0.6

solution_method = DirectMethod(problem, degree_control=3,
                               discretization_scheme = 'collocation',
                               degree=3, finite_elements=5, integrator_type='implicit')

x_sol, u_sol, V_sol = solution_method.solve()

x, y, u, t= solution_method.plot_simulate(x_sol, u_sol, [{'x':[0, 1]}, {'u':[0]}], 5, integrator_type ='implicit')
