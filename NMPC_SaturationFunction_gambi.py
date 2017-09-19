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
import random

SEED = 9946
random.seed(SEED)

N = 20
#PROBLEMATICS = [1,3,4,5,8,9,10,11,12,14,17,19]
PROBLEMATICS = []
state_constraints = True
#state_constraints = False
control_constraints = True
#control_constraints = False

times = []
obj = []
x_0 = []
for i in range(N):
    f = open('aug_lag.txt','w')
    f.write('iteration: '+`i`)
    f.close()
    x_0.append([2*(random.random()-0.5)*pi/6., 1*(random.random()-0.5), (random.random()-0.5), (random.random()-0.5)])
    if True: #not i in PROBLEMATICS:
        cost = casadi.SX.sym('cost')
        plant = PendulumCart() 
        model = PendulumCart() 
        #    problem = UpwardPendulumStabilization(model)
        problem = UpwardPendulumStabilization(model, 
    #                                          state_constraints = state_constraints, 
                                              control_constraints = control_constraints,
                                              t_f = 3.0,
                                              x_0 = DM(x_0[i])
                                              )
        plant.includeState(cost, casadi.mtimes(casadi.mtimes(plant.x_sym.T,problem.Q),plant.x_sym) + casadi.mtimes(casadi.mtimes(plant.u_sym.T,problem.R),plant.u_sym))
    
        r_eq_list = [(2, problem.model.x_sym[2] , 2, -2)] #problem.model.ode[3]
        
        ocp_solver = SaturationFunctionMethod(problem, r_eq_list, IndirectMethod, \
            {'degree': 1,},
                max_iter = 1, mu_0 = 1, beta= 10., mu_max = 1e3, finite_elements = 40, degree = 2, integrator_type = 'explicit',
                    )
        #
        dt = (problem.t_f - problem.t_0)/ocp_solver.finite_elements
        nmpc = NMPCScheme(plant, problem, ocp_solver, t_f = 10., dt = dt)
        X, U, T = nmpc.run()
        print 'Realized Objective: ', X[-1][4]
        
        #    
        nmpc.plot(X, U, [{'x':[0,1]},{'x':[2,3]},{'u':[0]}], T)
        times.append(nmpc.times)
        obj.append(X[-1][4].full()[0][0])

N = N-len(PROBLEMATICS)
print '====================================================='
print 'SATURATION FUNCTION, SEED:', SEED
print 'Average solution time: ', sum([sum(t[1:])/(len(t)-1) for t in times])/N
print 'Average first time: ', sum([t[0] for t in times])/N
print 'Average Objective: ', sum(obj)/N
print '====================================================='
##
#Total solution time:  80.1849999428
#First it. solution time:  4.4470000267
#Average solution time:  0.946724998951
