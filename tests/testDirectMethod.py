# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:55:27 2016

@author: marco
"""
import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))
from yaocptool.problems.cartpendulum import *
from yaocptool.methods import DirectMethod

model = PendulumCart()

control_constraints = False
state_constraints = True
state_constraints_2 = False

problem = UpwardPendulumStabilization(model, state_constraints=state_constraints,
                                      control_constraints=control_constraints)

solution_method = DirectMethod(problem, degree=3,
                               degree_control=3,
                               finite_elements=20,
                               integrator_type='implicit',
                               # discretization_scheme = 'collocation'
                               )

result = solution_method.solve()
result.plot([{'x': [0]}, {'x': [2, 3]},
             {'u': [0]}
             ])
# x, y, u, t= solution_method.plotSimulate(x_sol, u_sol, [{'x':[0]},{'x':[2,3]},{'u':[0]}], 5)
