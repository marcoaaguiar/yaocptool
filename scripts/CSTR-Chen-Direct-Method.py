import sys
from os.path import dirname, abspath

sys.path.append(abspath(dirname(dirname(__file__))))

from scripts.problems import create_CSTR_OCP
from yaocptool.methods import DirectMethod, AugmentedLagrangian

problem = create_CSTR_OCP()

# if False:
if True:
    solution_method_ind = DirectMethod(
        # solution_method_ind = IndirectMethod(
        problem, degree=4, degree_control=4,
        # solution_method = DirectMethod(problem, degree=3, degree_control=3,
        finite_elements=80,
        # integrator_type = 'explicit',
        # integrator_type='implicit',
        discretization_scheme='collocation',
    )
    result = solution_method_ind.solve()
    result.plot([{'x': [0, 1]}, {'x': [2, 3]},
                 # {'u':[0]},{'u':[1]},
                 # {'y':[3,4,5]}
                 {'y':[0,1,2]}
                 ])
    # time_dict = defaultdict(dict)
    # functions = defaultdict(dict)
    # for el in range(solution_method_ind.finite_elements):
    #     time_dict[el]['t_0'] = solution_method_ind.time_breakpoints[el]
    #     time_dict[el]['t_f'] = solution_method_ind.time_breakpoints[el + 1]
    #     time_dict[el]['y'] = [solution_method_ind.time_breakpoints[el] + solution_method_ind.delta_t * col for col in
    #                           solution_method_ind.collocation_points(solution_method_ind.degree)]
    #     functions[el]['y'] = Function('f', solution_method_ind.model.all_sym, [solution_method_ind.model.y_sym])
    # solution_method_ind.discretizer.get_system_at_given_times(result.x_interpolation_data['values'], result.y_interpolation_data['values'],
    #                                                           result.u_interpolation_data['values'], time_dict=time_dict,
    #                                                           functions=functions, p=result.p, theta=result.theta)
# if True:
if False:
    solution_method_al = AugmentedLagrangian(problem, DirectMethod,
                                             {'degree': 4, 'degree_control': 4,
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
