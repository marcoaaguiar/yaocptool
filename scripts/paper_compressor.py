from casadi import Function, vertcat

from scripts.problems import create_compressor
from yaocptool.methods import DirectMethod

model, problem = create_compressor()

solution_method_ind = DirectMethod(
    problem, degree=4, degree_control=1,
    finite_elements=20,
    discretization_scheme='collocation'
)

equilibrium = model.find_equilibrium([model.u_sym[1], model.x_sym[1] - 5.],
                                     guess=vertcat(problem.x_0,
                                                   vertcat([812.401, 5., 5., 0., 3.792884992501373]),
                                                   vertcat(812.401, .0)))
print(equilibrium)
problem.x_0 = equilibrium[0][:model.n_x]

result = solution_method_ind.solve()
result.plot([{'x': [0, 3]},
             {'x':[1], 'y': [1, 2, 3]},
             {'u':[1]},
             {'u':[0], 'y':[0]},
             ])

# model.simulate(x_0=problem.x_0, t_f=1, integrator_type='implicit')

f = Function('f_ode', [model.x_sym, model.y_sym, model.u_sym], [model.ode])
g = Function('f_alg', [model.x_sym, model.y_sym, model.u_sym], [model.alg])
