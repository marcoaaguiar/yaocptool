import sys
from os.path import dirname, abspath

sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cstr_chen import create_CSTR_OCP
from yaocptool.methods import DirectMethod, AugmentedLagrangian, IndirectMethod

problem = create_CSTR_OCP()

# if False:
if True:
    solution_method = IndirectMethod(problem, degree=3, degree_control=1,
    # solution_method = DirectMethod(problem, degree=3, degree_control=3,
                                   finite_elements=40,
                                   # integrator_type = 'explicit',
                                   # integrator_type='implicit',
                                   discretization_scheme='collocation',
                                   )

else:
    solution_method = AugmentedLagrangian(problem, DirectMethod,
                                          {'degree': 3, 'degree_control': 3,
                                         # 'integrator_type': 'implicit',
                                         # 'integrator_type': 'explicit',
                                         'discretization_scheme': 'collocation'
                                         },
                                          max_iter=3,
                                          mu_0=1,
                                          beta=10.,
                                          # relax_state_bounds=True,
                                        finite_elements=40,
                                          )

result = solution_method.solve()
result.plot([{'x': [0, 1]}, {'x': [2, 3]},
             # {'u':[0]},{'u':[1]},
             {'nu':[0,1,2]}
             ])
