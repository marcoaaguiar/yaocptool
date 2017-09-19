# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 19:01:00 2016

@author: marco
"""
import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cartpendulum import *
from yaocptool.methods import DirectMethod, IndirectMethod,  SaturationFunctionMethod



#control_constraints = True
control_constraints = False
state_constraints = True
#state_constraints = False
#state_constraints_2 = True
state_constraints_2 = False

###########
# Indirect 
##############

if True:
    model = PendulumCart()
    
    problem = UpwardPendulumStabilization(model, t_f  = 3)
#    problem.x_max[2] = 5
#    problem.x_min[2] = -5
    
    if state_constraints:
        eq_max = 2
        eq_min = -2
        r_eq_list = [(2, problem.model.x_sym[2] , eq_max, eq_min)] #problem.model.ode[3]
    if state_constraints_2:
        eq_max = 3
        eq_min = -3
        r_eq_list = [(2, problem.model.x_sym[2] , eq_max, eq_min)] #problem.model.ode[3]

    sat = SaturationFunctionMethod(problem, r_eq_list, IndirectMethod, \
        {'degree': 1},
            max_iter = 4, mu_0 = 10., beta= 10., finite_elements = 80, degree = 3, integrator_type = 'explicit')
    x_sol, u_sol, V_sol =sat.solve()
#    x_sol, u_sol, V_sol =sat.solve(V_sol)
    
    x, u, t= sat.plotSimulate(x_sol, u_sol, [{'x':[0,1]},{'x':[2,3]},{'u':[0,1]}], 5)


    from casadi import exp, Function
    xi = SX.sym('xi')

    psi = eq_max - (eq_max-eq_min)/(1+exp(4*xi/(eq_max-eq_min)))
    fpsi = Function('fpsi', [xi],[psi])
    psiv = map(lambda x: fpsi(x[4]),x)
    
    import matplotlib.pyplot as plt
    plt.plot(psiv)
    
    plt.plot([i[2] for i in x])