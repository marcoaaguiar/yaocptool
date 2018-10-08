# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:57:40 2016

@author: marco
"""
import sys
from os.path import dirname, abspath

from tests.models.cartpendulum import PendulumCart, UpwardPendulumStabilization, DownwardPendulumStabilization

sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.methods import IndirectMethod

model = PendulumCart()

problem = DownwardPendulumStabilization(model, t_f=3.)
# prob.x_max[2] = 5
# prob.x_min[2] = -5


indir_method = IndirectMethod(problem, degree=5, finite_elements=20, integrator_type='implicit',
                                initial_guess_heuristic='problem_info',
                              discretization_scheme='collocation')
solution = indir_method.solve()
solution.plot([{'x': [0, 1]}, {'x': [2, 3]}])

