import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cstr_chen import create_CSTR_OCP
from yaocptool.methods import DirectMethod, AugmentedLagrange, IndirectMethod



problem = create_CSTR_OCP()

# solution_method = DirectMethod(problem, degree = 3, degree_control = 1,
#                           finite_elements = 40,
#                           # integrator_type = 'explicit',
#                           integrator_type = 'implicit',
#                           discretization_method = 'collocation',
#                           )
solution_method = AugmentedLagrange(problem, DirectMethod, \
                        {'degree': 3, 'degree_control':3,
                         # 'discretization_method': 'collocation'
                        },
                        max_iter=3,
                        mu_0=1.,
                        beta=10.,
                        # relax_state_bounds=True,
                        finite_elements=40,
                        integrator_type='implicit'
                                    )

x_sol, u_sol, V_sol = solution_method.solve()
x, y, u, t= solution_method.plotSimulate(x_sol, u_sol, [{'x':[0,1]},{'x':[2,3]},{'u':[0]},{'u':[1]}], 1, integrator_type='explicit')
# u = problem.u_ref
# u = [25, -4000]
# u[0] *=3600
# u[1] *=100
# sim = problem.model.simulateInterval(x_0 = problem.x_0, t_grid=[x /3600. for x in range(1, 2000)], t_f = 1./36000, t_0 =0, p = u)
#
# for s in sim[0]:
#     print s