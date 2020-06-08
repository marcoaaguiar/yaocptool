# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:55:27 2016

@author: marco
"""
import sys
from os.path import dirname, abspath

from manual_tests.models.linear_models import MIMO2x2DAE, StabilizationMIMO2x2WithInequality
from yaocptool.methods import DirectMethod, AugmentedLagrangian, IndirectMethod

sys.path.append(abspath(dirname(dirname(__file__))))

# Settings:
AUG_LAGRANGIAN = True
INDIRECT = True
DIRECT = True

# discretization_scheme='multiple-shooting'
discretization_scheme = 'collocation'
degree = 3
degree_control = 3
finite_elements = 60

figs = []

if AUG_LAGRANGIAN:
    model = MIMO2x2DAE('aug_lag_1')
    problem = StabilizationMIMO2x2WithInequality(model, t_f=10)
    ocp_solver_class = DirectMethod
    ocp_solver_opts = dict(
        # integrator_type='implicit',
        discretization_scheme=discretization_scheme)

    solution_method = AugmentedLagrangian(problem,
                                          ocp_solver_class=ocp_solver_class,
                                          solver_options=ocp_solver_opts,
                                          degree=degree,
                                          degree_control=degree_control,
                                          finite_elements=finite_elements,
                                          max_iter=5,
                                          mu_max=1e6)

    result = solution_method.solve(p=[1], x_0=[2, 3, 0])
    figs = result.plot(
        [  # {'x': [0]},
            {
                'x': [0, 1]
            }, {
                'u': [0]
            }, {
                'nu': [0, 1]
            }
        ],
        show=False)

if INDIRECT:
    model = MIMO2x2DAE('indirect')
    problem = StabilizationMIMO2x2WithInequality(model, t_f=10)
    solution_method2 = IndirectMethod(
        problem,
        degree=degree,
        degree_control=degree_control,
        finite_elements=finite_elements,
        discretization_scheme=discretization_scheme)
    result2 = solution_method2.solve(p=[1], x_0=[2, 3, 0, 0])
    figs = result2.plot([{
        'x': [0, 1]
    }, {
        'u': 'all'
    }, {
        'y': [2, 3]
    }],
                        figures=figs,
                        show=True)

if DIRECT:
    model = MIMO2x2DAE('direct')
    problem = StabilizationMIMO2x2WithInequality(model, t_f=10)
    solution_method2 = DirectMethod(
        problem,
        degree=degree,
        degree_control=degree_control,
        finite_elements=finite_elements,
        discretization_scheme=discretization_scheme)
    result2 = solution_method2.solve(p=[1], x_0=[2, 3, 0])
    figs = result2.plot(
        [  # {'x': [0]},
            {
                'x': [0, 1]
            }, {
                'u': [0]
            }
        ],
        figures=figs)
