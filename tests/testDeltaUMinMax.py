# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:55:27 2016

@author: marco
"""
import sys
from os.path import dirname, abspath

from casadi import vertcat, DM, mtimes

from yaocptool.methods import DirectMethod
from yaocptool.modelling import SystemModel, OptimalControlProblem

sys.path.append(abspath(dirname(dirname(__file__))))

model = SystemModel()
x = model.create_state('x', 2)
u = model.create_control('u', 2)

a = DM([[-1, -2], [5, -1]])
# b = DM([1, 0])
b = DM([[1, 0], [0, 1]])

model.include_system_equations(ode=vertcat(mtimes(a, x) + mtimes(b, u)))

problem = OptimalControlProblem(model, obj={'Q': DM.eye(2), 'R': DM.eye(2)}, x_0=[1, 1],
                                )

problem.delta_u_max = [0.05, 0.05]
problem.delta_u_min = [-0.05, -0.05]
problem.last_u = [0,0]

solution_method = DirectMethod(problem, degree=3,
                               degree_control=1,
                               finite_elements=20,
                               # integrator_type='implicit',
                               # discretization_scheme = 'collocation'
                               )


result = solution_method.solve()

result.plot([  # {'x': [0]},
    {'x': [0, 1]},
    {'u': [0, 1]}
])

result = solution_method.solve(last_u = [0,1])
result.plot([  # {'x': [0]},
    {'x': [0, 1]},
    {'u': [0, 1]}
])

