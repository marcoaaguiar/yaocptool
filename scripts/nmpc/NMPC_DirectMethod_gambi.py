# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Thu Nov 17 14:45:29 2016

@author: marco
"""

import sys
from os.path import dirname, abspath

sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.methods import DirectMethod
from yaocptool.nmpc import NMPCScheme
from yaocptool import casadi
import random

random.seed(9946)

N = 20

state_constraints = True
# state_constraints = False
control_constraints = True
# control_constraints = False

times = []
obj = []
x_0 = []
for i in range(N):
    x_0.append([2 * (random.random() - 0.5) * pi / 6., 1 * (random.random() - 0.5), (random.random() - 0.5),
                (random.random() - 0.5)])
    cost = casadi.SX.sym('cost')
    plant = PendulumCart()
    model = PendulumCart()
    problem = UpwardPendulumStabilization(model,
                                          state_constraints=state_constraints,
                                          control_constraints=control_constraints,
                                          t_f=3.0,
                                          x_0=DM(x_0[i])
                                          )
    plant.include_state(cost, casadi.mtimes(casadi.mtimes(plant.x_sym.T, problem.Q), plant.x_sym) + casadi.mtimes(
        casadi.mtimes(plant.u_sym.T, problem.R), plant.u_sym))
    ocp_solver = DirectMethod(problem, degree=1, finite_elements=40, integrator_type='explicit')

    dt = (problem.t_f - problem.t_0) / ocp_solver.finite_elements
    nmpc = NMPCScheme(plant, problem, ocp_solver, t_f=10., dt=dt)
    X, U, T = nmpc.run
    #    
    print('Realized Objective: ', X[-1][4])
    nmpc.plot(X, U, [{'x': [0, 2]}, {'x': [4]}, {'u': [0]}], T)
    times.append(nmpc.times)
    obj.append(X[-1][4].full()[0][0])

print('=====================================================')
print('DIRECT MULTIPLE-SHOOTING, SEED:'.SEED)
print('Average solution time: ', sum([sum(t[1:]) / (len(t) - 1) for t in times]) / N)
print('Average first time: ', sum([t[0] for t in times]) / N)
print('Average Objective: ', sum(obj) / N)
print('=====================================================')

dir_times = times
dir_obj = obj
dir_x_0 = x_0
##
# Total solution time:  6.52900004387
# First it. solution time:  0.197000026703
# Average solution time:  0.0816125005484

##
# Total solution time:  9.46799993515
# First it. solution time:  0.381999969482
# Average solution time:  0.113574999571

# X_mat = [dm.full() for dm in X]
# U_mat = [dm.full() for dm in U]
# import numpy, scipy.io
# scipy.io.savemat('matlab/dir_method.mat', mdict={'X': X_mat, 'U': U_mat, 'T':T})
