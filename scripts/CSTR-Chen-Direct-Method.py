import sys
from os.path import dirname, abspath

sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cstr_chen import create_CSTR_OCP
from yaocptool.methods import DirectMethod, AugmentedLagrangian, IndirectMethod

problem = create_CSTR_OCP()

if False:
# if True:
    solution_method_ind = IndirectMethod(problem, degree=5, degree_control=5,
                                     # solution_method = DirectMethod(problem, degree=3, degree_control=3,
                                     finite_elements=80,
                                     # integrator_type = 'explicit',
                                     # integrator_type='implicit',
                                     discretization_scheme='collocation',
                                     )
    result = solution_method_ind.solve()
    result.plot([{'x': [0, 1]}, {'x': [2, 3]},
                 # {'u':[0]},{'u':[1]},
                 {'y':[3,4,5]}
                 ])
if True:
    solution_method_al = AugmentedLagrangian(problem, DirectMethod,
                                          {'degree': 5, 'degree_control': 5,
                                           # 'integrator_type': 'implicit',
                                           # 'integrator_type': 'explicit',
                                           'discretization_scheme': 'collocation'
                                           },
                                          max_iter=3,
                                          mu_0=1,
                                          beta=10.,
                                          # relax_state_bounds=True,
                                          finite_elements=80,
                                          )
    result = solution_method_al.solve()
    result.plot([{'x': [0, 1]}, {'x': [2, 3]},
                 # {'u':[0]},{'u':[1]},
                 {'nu': [0, 1, 2]}
                 ])