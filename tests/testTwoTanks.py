# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Wed Nov 02 18:57:40 2016

@author: marco
"""


from sys import path
from os.path import dirname, abspath

path.append(dirname(dirname(abspath(__file__))))

from yaocptool.problems.twotanks import *
from yaocptool.methods import IndirectMethod, DirectMethod, AugmentedLagrangian
import time

# model = TwoTanks()

problem = StabilizationTwoTanks()
# problem = StabilizationTwoTanksCentralized()
# prob.x_max[2] = 5
# prob.x_min[2] = -5
# problem.V = 16
# problem.t_f = 1
#
# solution_method = IndirectMethod(problem,
# solution_method = DirectMethod(problem, degree_control=5,
#                                                               discretization_scheme = 'collocation',
                               # degree=5, finite_elements=20, integrator_type='implicit')
##
t1 = time.time()

solution_method = AugmentedLagrangian(problem, IndirectMethod,
                                      {},  #{ 'discretization_scheme': 'multiple-shooting'},
        degree = 3, degree_control = 3,
                                      relax_algebraic = True, relax_external_algebraic = True, relax_connecting_equations = False,
                                      max_iter = 2, mu_0 = 1, beta= 10., finite_elements = 6, integrator_type = 'implicit')

x_sol, u_sol, V_sol = solution_method.solve()
print(time.time() - t1)

x, y, u, t = solution_method.plot_simulate(x_sol, u_sol, [{'x': [0, 1]}, {'x': [2]}, {'u': [0]}], 10,
                                           integrator_type='implicit')

