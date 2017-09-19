# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:45:29 2016

@author: marco
"""

import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cartpendulum import *
from yaocptool.methods import DirectMethod, IndirectMethod, SaturationFunctionMethod
from yaocptool import NMPCScheme
from yaocptool import casadi

cost = casadi.SX.sym('cost')
plant = PendulumCart() 
model = PendulumCart() 
#    problem = UpwardPendulumStabilization(model)
problem = UpwardPendulumStabilization(model, 
#                                      state_constraints = False, 
                                      control_constraints = True,
                                      t_f = 3.0,
#                                      x_0 = DM([pi-1, 0, 0, 0]),
                                      )

r_eq_list = [(2, problem.model.x_sym[2] , 2, -2)] #problem.model.ode[3]
plant.includeState(cost, casadi.mtimes(casadi.mtimes(plant.x_sym.T,problem.Q),plant.x_sym) + casadi.mtimes(casadi.mtimes(plant.u_sym.T,problem.R),plant.u_sym))

ocp_solver = SaturationFunctionMethod(problem, r_eq_list, IndirectMethod, \
    {'degree': 1,},
#            max_iter = 1, mu_0 = 1., beta= 2., mu_max = 1e4, finite_elements = 20, degree = 2, integrator_type = 'explicit'
#        max_iter = 1, mu_0 = 10., beta= 5., mu_max = 1e4, finite_elements = 20, degree = 1, integrator_type = 'explicit',
            max_iter = 1, mu_0 = 1, beta= 10., mu_max = 1e3, finite_elements = 40, degree = 2, integrator_type = 'explicit',
            )
#
dt = (problem.t_f - problem.t_0)/ocp_solver.finite_elements
nmpc = NMPCScheme(plant, problem, ocp_solver, t_f = 10., dt = dt)

X, U, T = nmpc.run()
print 'Realized Objective: ', X[-1][4]

#    
nmpc.plot(X, U, [{'x':[0]},{'x':[2]},{'u':[0]}], T)

##
#Total solution time:  80.1849999428
#First it. solution time:  4.4470000267
#Average solution time:  0.946724998951
