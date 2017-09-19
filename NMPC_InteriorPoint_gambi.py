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
import random

SEED = 9946
random.seed(SEED)

N = 20

state_constraints = True
#state_constraints = False
control_constraints = True
#control_constraints = False

times = []
obj = []
x_0 = []
for i in range(N):
    x_0.append([2*(random.random()-0.5)*pi/6., 1*(random.random()-0.5), (random.random()-0.5), (random.random()-0.5)])
    cost = casadi.SX.sym('cost')
    plant = PendulumCart() 
    model = PendulumCart() 
    #    problem = UpwardPendulumStabilization(model)
    problem = UpwardPendulumStabilization(model, 
                                          state_constraints = state_constraints, 
                                          control_constraints = control_constraints,
                                          t_f = 3.0,
                                          x_0 = DM(x_0[i])
                                          )
    plant.includeState(cost, casadi.mtimes(casadi.mtimes(plant.x_sym.T,problem.Q),plant.x_sym) + casadi.mtimes(casadi.mtimes(plant.u_sym.T,problem.R),plant.u_sym))
    
    ocp_solver = InteriorPoint(problem, IndirectMethod, \
        {'degree': 1,},
            max_iter = 1, mu_0 = 1., beta = 5.,  finite_elements = 40,  integrator_type = 'explicit'
            )    
    #
    dt = (problem.t_f - problem.t_0)/ocp_solver.finite_elements
    nmpc = NMPCScheme(plant, problem, ocp_solver, t_f = 10., dt = dt)
    X, U, T = nmpc.run()
    print 'Realized Objective: ', X[-1][4]
    
    #    
    nmpc.plot(X, U, [{'x':[0,2]},{'x':[2]},{'u':[0]}], T)
    times.append(nmpc.times)
    obj.append(X[-1][4].full()[0][0])

print '====================================================='
print 'INTERIOR POINT, SEED:', SEED
print 'Average solution time: ', sum([sum(t[1:])/(len(t)-1) for t in times])/N
print 'Average first time: ', sum([t[0] for t in times])/N
print 'Average Objective: ', sum(obj)/N
print '====================================================='
