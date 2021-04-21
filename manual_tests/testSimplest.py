from yaocptool.methods import DirectMethod
from yaocptool.modelling import OptimalControlProblem, SystemModel
from yaocptool.util.util import Timer

model = SystemModel()
x = model.create_state("x")
u = model.create_control("u")

model.include_equations(ode=(-x + u))

problem = OptimalControlProblem(model, obj={"Q": 1, "R": 1}, x_0=[1], t_f=5.0)
# problem.u_min[0] = 0
# problem.u_max[0] = 1

# problem.x_min[0] = 0.6

solution_method = DirectMethod(
    problem,
    #  discretization_scheme="multiple-shooting",
    discretization_scheme="collocation",
    degree=3,
    degree_control=1,
    finite_elements=20,
    integrator_type="implicit",
)

with Timer(verbose=True) as once_timer:
    solution = solution_method.solve()

df = solution.to_dataset().as_dataframe()
df.plot()
#  solution.plot({"x": [0, 1]}, {"u": [0]})
