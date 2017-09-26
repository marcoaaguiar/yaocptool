# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 19:01:00 2016

@author: marco
"""
import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cartpendulum import *
from yaocptool.methods import DirectMethod, IndirectMethod, AugmentedLagrange



#solve_indirect = False
solve_indirect = True
#solve_direct = True
solve_direct = False

for solution_method_class in [DirectMethod, IndirectMethod]:
    for discretization in ['collocation', 'multiple-shooting']:
        if discretization == 'multiple-shooting':
            for integrator_type in ['explicit', 'implicit']:
                model = PendulumCart()

                problem = UpwardPendulumStabilization(model,
                                                      state_constraints = True,
                                                      t_f = 3.
                                                      )

                solution_method = AugmentedLagrange(problem, solution_method_class, \
                    {'degree': 4, 'degree_control':4, 'discretization_scheme':discretization},
                        max_iter = 5, mu_0 = 10, beta= 10.,
                        relax_state_bounds = True,
                        finite_elements = 40,
                                        integrator_type = 'explicit')

                result =solution_method.solve()
                result.plot([{'x':[0]}, {'x':[2],}
                             # {'u':[0]},
                             # {'u':[1]}
                             ])
                # x, y, u, t= aug.plotSimulate(x_sol, u_sol, [{'x':[0]},{'u':[0]},{'x':[2],'u':[1]}], 1,integrator_type= 'explicit')
