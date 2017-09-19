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

#control_constraints = True
control_constraints = False
#state_constraints = True
state_constraints = True
#state_constraints_2 = True
state_constraints_2 = False

problem = UpwardPendulumStabilization(model, state_constraints= state_constraints, 
                                      control_constraints = control_constraints)
#problem.t_f = 1
#problem.u_max[0] = 5
#problem.u_min[0] = -5

dir_method = DirectMethod(problem, degree = 1, degree_control = 1, finite_elements = 40, integrator_type = 'implicit')
x_sol, u_sol, V_sol = dir_method.solve()
x, y, u, t= dir_method.plotSimulate(x_sol, u_sol, [{'x':[0]},{'x':[2,3]},{'u':[0]}], 5)

