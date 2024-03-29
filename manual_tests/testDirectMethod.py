# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:55:27 2016

@author: marco
"""
import sys
from os.path import abspath, dirname

from manual_tests.models.linear_models import MIMO2x2, StabilizationMIMO2x2
from manual_tests.models.vanderpol import VanDerPol, VanDerPolStabilization
from yaocptool.methods import DirectMethod

sys.path.append(abspath(dirname(dirname(__file__))))

# model = VanDerPol()
# problem = VanDerPolStabilization(model)

model = MIMO2x2()
problem = StabilizationMIMO2x2(model)

for discretization_scheme in ["collocation", "multiple-shooting"][1:]:
    solution_method = DirectMethod(
        problem,
        degree=3,
        degree_control=3,
        finite_elements=20,
        integrator_type="implicit",
        discretization_scheme=discretization_scheme,
    )

    result = solution_method.solve()
    result.plot([{"x": "all"}, {"u": "all"}])  # {'x': [0]},
# x, y, u, t= solution_method.plot_simulate(x_sol, u_sol, [{'x':[0]},{'x':[2,3]},{'u':[0]}], 5)
