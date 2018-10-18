# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:57:40 2016

@author: marco
"""
from __future__ import print_function

from tests.models.twotanks import TwoTanks, StabilizationTwoTanks
from yaocptool.methods import IndirectMethod, AugmentedLagrangian
import time

model = TwoTanks()

problem = StabilizationTwoTanks()
# problem = StabilizationTwoTanksCentralized()
# prob.x_max[2] = 5
# prob.x_min[2] = -5
# problem.V = 16
# problem.t_f = 1
#
# solution_method = DirectMethod(problem, degree_control=5,
#                                                               discretization_scheme = 'collocation',
# degree=5, finite_elements=20, integrator_type='implicit')
##

solution_method = IndirectMethod(problem,
                                 discretization_scheme='collocation',
                                 degree=3,
                                 finite_elements=20)

solution = solution_method.solve()

solution.plot([{'x': [0, 1]}, {'x': [2]}])
