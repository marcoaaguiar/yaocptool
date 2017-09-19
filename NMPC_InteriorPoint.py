# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:45:29 2016

@author: marco
"""

import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cartpendulum import *
from yaocptool.methods import DirectMethod, IndirectMethod, InteriorPoint
from yaocptool import NMPCScheme
from yaocptool import casadi

cost = casadi.SX.sym('cost')
plant = PendulumCart() 
model = PendulumCart() 
#    problem = UpwardPendulumStabilization(model)
problem = UpwardPendulumStabilization(model, 
                                      state_constraints = True, 
#                                      control_constraints = True,
                                      t_f = 3.0,
#                                      x_0 = DM([pi-1, 0, 0, 0]),
                                      )
plant.includeState(cost, casadi.mtimes(casadi.mtimes(plant.x_sym.T,problem.Q),plant.x_sym) + casadi.mtimes(casadi.mtimes(plant.u_sym.T,problem.R),plant.u_sym))

ocp_solver = InteriorPoint(problem, IndirectMethod, \
    {'degree': 1,},
        max_iter = 1, mu_0 = .1, beta = 6.,  finite_elements = 40,  integrator_type = 'explicit', mu_min = 1e-5
        )    
#
dt = (problem.t_f - problem.t_0)/ocp_solver.finite_elements
nmpc = NMPCScheme(plant, problem, ocp_solver, t_f = 10., dt = dt)
X, U, T = nmpc.run()
print 'Realized Objective: ', X[-1][4]

#    
nmpc.plot(X, U, [{'x':[0,2]},{'x':[2]},{'u':[0]}], T)

##
#Total solution time:  63.1879999638
#First it. solution time:  3.35099983215
#Average solution time:  0.747962501645

#
#Total solution time:  69.1499998569
#First it. solution time:  2.21199989319
#Average solution time:  0.836724999547
