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

state_constraints = True
#state_constraints = False
control_constraints = True
#control_constraints = False

#plant = PendulumCart(l = 1) 
cost = casadi.SX.sym('cost')
plant = PendulumCart()
model = PendulumCart() 
problem = UpwardPendulumStabilization(model, 
                                      state_constraints = state_constraints, 
                                      control_constraints = control_constraints, 
                                      t_f = 3.0,
#                                      x_0 = DM([pi-1, 0, 0, 0])
                                      )

plant.includeState(cost, casadi.mtimes(casadi.mtimes(plant.x_sym.T,problem.Q),plant.x_sym) + casadi.mtimes(casadi.mtimes(plant.u_sym.T,problem.R),plant.u_sym))
ocp_solver = AugmentedLagrange(problem, IndirectMethod, \
    {'degree': 1,},
        max_iter = 1, mu_0 = 1e4, beta= 10., mu_max = 1e4, finite_elements = 40, degree = 2, integrator_type = 'explicit',
        relax_state_bounds = True,
        )    


dt = (problem.t_f - problem.t_0)/ocp_solver.finite_elements
nmpc = NMPCScheme(plant, problem, ocp_solver, t_f = 10., dt = dt)


X, U, T = nmpc.run()
print 'Realized Objective: ', X[-1][4]
#    
nmpc.plot(X, U, [{'x':[0,1]},{'x':[4]},{'u':[0]}], T)

##
#Solution time:  0.0970001220703
#Total solution time:  12.6100001335
#First it. solution time:  1.27900004387
#Average solution time:  0.143430380882

X_mat = [dm.full() for dm in X]
U_mat = [dm.full() for dm in U]
import numpy, scipy.io
scipy.io.savemat('matlab/aug_lagrange.mat', mdict={'X': X_mat, 'U': U_mat, 'T':T})

