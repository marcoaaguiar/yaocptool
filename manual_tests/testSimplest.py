from yaocptool.util.util import Timer
from yaocptool.methods import IndirectMethod
from yaocptool.modelling import SystemModel, OptimalControlProblem

model = SystemModel()
x = model.create_state("x")
u = model.create_control("u")

model.include_equations(ode=(-x + u))

problem = OptimalControlProblem(model, obj={"Q": 1, "R": 1}, x_0=[1], t_f=5.0)
# problem.u_min[0] = 0
# problem.u_max[0] = 1

# problem.x_min[0] = 0.6

solution_method = IndirectMethod(
    problem,
    discretization_scheme="multiple-shooting",
    # discretization_scheme='collocation',
    degree=3,
    finite_elements=30,
    integrator_type="implicit",
)

with Timer(verbose=True) as once_timer:
    solution = solution_method.solve()

solution.plot({"x": [0, 1]}, {"u": [0]})
