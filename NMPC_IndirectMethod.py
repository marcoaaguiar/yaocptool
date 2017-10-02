# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:45:29 2016

@author: marco
"""

import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cartpendulum import *
from yaocptool.methods import DirectMethod, IndirectMethod, AugmentedLagrange
from yaocptool import NMPCScheme
from yaocptool import casadi

cost = casadi.SX.sym('cost')
plant = PendulumCart() 
model = PendulumCart() 
state_constraints = False
state_constraints = False
problem = UpwardPendulumStabilization(model,                                       
#                                      state_constraints = True, 
#                                      control_constraints = True,
                                      t_f =3.)
plant.include_state(cost, casadi.mtimes(casadi.mtimes(plant.x_sym.T, problem.Q), plant.x_sym) + casadi.mtimes(casadi.mtimes(plant.u_sym.T, problem.R), plant.u_sym))
ocp_solver = IndirectMethod(problem, degree = 1, finite_elements = 40, 
                            integrator_type = 'explicit')

dt = (problem.t_f - problem.t_0)/ocp_solver.finite_elements
nmpc = NMPCScheme(plant, problem, ocp_solver, t_f = 10., dt = dt)
X, U, T = nmpc.run()
print 'Realized Objective: ', X[-1][4]

#    
nmpc.plot(X, U, [{'x':[0]},{'x':[2]},{'u':[0]}], T)

##
#Total solution time:  4.70300006866
#First it. solution time:  0.363000154495
#Average solution time:  0.0587875008583
