# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 19:01:00 2016

@author: marco
"""
import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.toysys import *
from yaocptool.methods import DirectMethod, IndirectMethod,  SaturationFunctionMethod



#solve_indirect = False
solve_indirect = True
#solve_direct = True
solve_direct = False

###########
# Indirect 
##############

if solve_indirect:
    model = ToySystem()
    
    problem = ToyProblem(model, t_f  = 5)
#    problem.x_max[2] = 5
#    problem.x_min[2] = -5
    
    eq_max = .2
    eq_min = -2
    
    r_eq_list = [(2, problem.model.x_sym[0] , eq_max, eq_min)] #problem.model.ode[3]
    sat = SaturationFunctionMethod(problem, r_eq_list, IndirectMethod, \
        {'degree': 1},
            max_iter = 4, mu_0 = 10., beta= 10., finite_elements = 80, degree = 4, integrator_type = 'explicit', parametrize = True)
    x_sol, u_sol, V_sol =sat.solve()
#    x_sol, u_sol, V_sol =sat.solve(V_sol)
    
    x, u, t= sat.plotSimulate(x_sol, u_sol, [{'x':[0,1]},{'x':[2]},{'u':[1]}], 5)

    from casadi import exp, Function, gradient
    xi = SX.sym('xi')

    psi = eq_max - (eq_max-eq_min)/(1+exp(4*xi/(eq_max-eq_min)))
    fpsi = Function('fpsi', [xi],[psi])
    fdpsi = Function('fpsi', [xi],[gradient(psi,xi)])
    psiv = map(lambda x: fpsi(x[2]),x)
    dpsiv = map(lambda x: fdpsi(x[2]),x)
    
    import matplotlib.pyplot as plt
    plt.plot(psiv)
    plt.plot(dpsiv)
    
    plt.plot([i[0] for i in x])

