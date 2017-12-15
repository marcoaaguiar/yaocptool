# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 19:01:00 2016

@author: marco
"""
# import sys
# from os.path import dirname, abspath
# sys.path.append(abspath(dirname(dirname(__file__))))
from yaocptool.methods import DirectMethod, IndirectMethod, AugmentedLagrangian

# solve_indirect = False
solve_indirect = True
# solve_direct = True
solve_direct = False

k = 0
result_list = []
for solution_method_class in [DirectMethod, IndirectMethod]:  # , IndirectMethod]:
    for discretization in ['collocation']:  # 'collocation', 'multiple-shooting']:
    # for discretization in ['collocation', 'multiple-shooting']:  # 'collocation', 'multiple-shooting']:
        for integrator_type in ['explicit', 'implicit']:
            if discretization == 'collocation' and integrator_type == 'implicit':
                pass
            else:
                print(
                'class: {}, discretization: {}, integrator_type: {}'.format(solution_method_class.__name__,
                                                                            discretization, integrator_type))
                model = PendulumCart()

                problem = UpwardPendulumStabilization(model,
                                                      state_constraints=True,
                                                      t_f=3.
                                                      )

                solution_method = AugmentedLagrangian(problem, solution_method_class,
                                                      {'degree': 4, 'degree_control': 4,
                                                     'discretization_scheme': discretization,
                                                     'nlpsol_opts': {
                                                         'ipopt.print_level': 0,
                                                         'print_time': False
                                                     }
                                                     },
                                                      max_iter=5, mu_0=10, beta=10.,
                                                      relax_state_bounds=True,
                                                      finite_elements=40,
                                                      integrator_type=integrator_type)

                result = solution_method.solve()
                result.plot([{'x': [0]}, {'x': [2], }
                             # {'u':[0]},
                             # {'u':[1]}
                             ])
                result_list.append(result)
                k += 1
obj = None
for result in result_list:
    if result.method_name == 'DirectMethod':
        if obj is None:
            obj = result.objective
        else:
            if abs(obj - result.objective) < 1e-6:
                print('wroooong', result.objective)
