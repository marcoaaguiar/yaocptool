# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 19:01:00 2016

@author: marco
"""
import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cartpendulum import *
from yaocptool.methods import DirectMethod, IndirectMethod, InteriorPoint



solve_indirect = False
solve_indirect = True
solve_direct = False
#solve_direct = True

###########
# Indirect 
##############

if solve_indirect:
    model = PendulumCart()
    
    problem = UpwardPendulumStabilization(model, state_constraints = True, 
#                                              control_constraints = True, 
                                          t_f = 3.)
#    problem.x_max[2] = 2
#    problem.x_min[2] = -2
    
    intp = InteriorPoint(problem, IndirectMethod, \
        {'degree': 1, 'finite_elements': 40, 'integrator_type': 'explicit'},
            relax_control_bounds = False, 
            max_iter = 8, 
            mu_0 = 1., 
            beta = 10., 
            mu_min = 1e-3)
    x_sol, u_sol =intp.solve()
    
    x, u, t= intp.ocp_solver.plotSimulate(x_sol, u_sol, [{'x':[0,2]},{'u':[0]}], 1, p = intp.mu)


#################3
# Direct 
############3

if solve_direct:
    model = PendulumCart()
    
    problem = UpwardPendulumStabilization(model, state_constraints = True)
#    problem.x_max[2] = 10
#    problem.x_min[2] = -10
    intp = InteriorPoint(problem, DirectMethod,  
        {'degree': 1, 'finite_elements': 40, 'integrator_type': 'implicit'},
            relax_control_bounds = True, max_iter = 4, mu_0 = 10)
    x_sol, u_sol =intp.solve()
    x, u, t= intp.ocp_solver.plotSimulate(x_sol, u_sol, [{'x':[0,2]},{'x':[4]},{'u':[0]}], 5)
