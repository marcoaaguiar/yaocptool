# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:57:40 2016

@author: marco
"""
from manual_tests.models.linear_models import MIMO2x2, StabilizationMIMO2x2
from yaocptool.methods import IndirectMethod

# model = VanDerPol()
# problem = VanDerPolStabilization(model)

model = MIMO2x2()
problem = StabilizationMIMO2x2(model)

indir_method = IndirectMethod(
    problem,
    degree=3,
    finite_elements=20,
    # integrator_type='implicit',
    # initial_guess_heuristic='problem_info',
    # discretization_scheme='multiple-shooting',
    discretization_scheme='collocation')
solution = indir_method.solve()
solution.plot([{'x': 'all'}, {'u': 'all'}])
