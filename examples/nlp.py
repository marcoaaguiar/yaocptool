from casadi import MX

from yaocptool.optimization import NonlinearOptimizationProblem

nlp = NonlinearOptimizationProblem(name='example')

# create the nlp variables
x = nlp.create_variable('x')
# include lower/upper bound on creation
y = nlp.create_variable('y', ub=10, lb=-30)

# the nlp can be parametrized
# center = nlp.create_parameter('center_point', size=2)
center = [5, 10]

# set the objective function
nlp.set_objective((x - center[0]) ** 2 + (y - center[1]) ** 2)

# include a constraint by passing it mathematically
# note: casadi/python does not allow for double bounded constraint (e.g.: 1 <= x + y <=3)
nlp.include_constraint(x ** 2 + y ** 2 <= 3)

# optionally constraints can be included with
nlp.include_equality(x + y, 0)  # x + u = 0
nlp.include_inequality(x + y, -5, 5)  # -5 <= x+y <= 5

# solves the problem
print(nlp.solve(initial_guess=[0, 1]))
