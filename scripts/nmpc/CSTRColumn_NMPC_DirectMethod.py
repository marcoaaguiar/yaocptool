# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:37:54 2017

@author: marco
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:45:29 2016

@author: marco
"""

import sys
from os.path import dirname, abspath

sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cstrcolumn import *
from yaocptool.methods import DirectMethod
from yaocptool.nmpc import NMPCScheme

# plant = PendulumCart(l = 9.8/9*1.05)
plant = CSTRColumnSystem()
model = CSTRColumnSystem()
problem = CSTRColumnProblem(model,
                            state_constraints=False,
                            control_constraints=False
                            #                                      t_f = 3.0,
                            #                                      x_0 = DM([pi-1, 0, 0, 0]),
                            )
# problem.x_min[2] = -2.5
# problem.x_max[2] = 2.5

ocp_solver = DirectMethod(problem, degree=1, finite_elements=20, integrator_type='implicit')

dt = (problem.t_f - problem.t_0) / ocp_solver.finite_elements
nmpc = NMPCScheme(plant, problem, ocp_solver, t_f=10., dt=dt)
X, U, T = nmpc.run
#    
nmpc.plot(X, U, [{'x': [0, 2]}, {'x': [2]}, {'u': [0]}], T)

##
# Total solution time:  6.52900004387
# First it. solution time:  0.197000026703
# Average solution time:  0.0816125005484

##
# Total solution time:  9.46799993515
# First it. solution time:  0.381999969482
# Average solution time:  0.113574999571

X_mat = [dm.full() for dm in X]
U_mat = [dm.full() for dm in U]
import scipy.io

scipy.io.savemat('matlab/dir_method.mat', mdict={'X': X_mat, 'U': U_mat, 'T': T})
