# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 19:01:00 2016

@author: marco
"""
import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cartpendulum import *
from yaocptool.methods import DirectMethod, IndirectMethod, AugmentedLagrange



#solve_indirect = False
solve_indirect = True
#solve_direct = True
solve_direct = False

###########
# Indirect 
##############

if solve_indirect:
    model = PendulumCart()
    
    problem = UpwardPendulumStabilization(model, 
                                          state_constraints = True, 
#                                          control_constraints = True,
                                          t_f = 3.
                                          )
#    problem.x_max[2] = 2
#    problem.x_min[2] = -2
    
    aug = AugmentedLagrange(problem, IndirectMethod, \
        {'degree': 5,},
            max_iter = 5, mu_0 = 10, beta= 10., 
            relax_state_bounds = True,
            finite_elements = 40,  integrator_type = 'explicit')
    
#    aug.createCalculateNewNu()

    x_sol, u_sol, V_sol =aug.solve()
    
    x, y, u, t= aug.plotSimulate(x_sol, u_sol, [{'x':[0]},{'u':[0]},{'x':[2],'u':[1]}], 1,integrator_type= 'explicit')
