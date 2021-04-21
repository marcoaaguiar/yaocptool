# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:57:40 2016

@author: marco
"""
from __future__ import print_function

import time

from manual_tests.models.twotanks import StabilizationTwoTanks, TwoTanks
from yaocptool.methods import AugmentedLagrangian, DirectMethod, IndirectMethod

model = TwoTanks()

problem = StabilizationTwoTanks(model)

solution_method = DirectMethod(
    problem,
    # discretization_scheme='collocation',
    degree=3,
    finite_elements=20,
)

solution = solution_method.solve()
solution.plot([{"x": "all"}, {"y": "all"}, {"u": "all"}])
