from __future__ import print_function

from yaocptool.methods import DirectMethod
from yaocptool.modelling import OptimalControlProblem, SystemModel

# PART 1
model = SystemModel(name="simple_model")
x = model.create_state("x")  # vector of state variables
u = model.create_control("u")  # vector of control variables

# Include the dynamic equation
ode = [-x + u]
model.include_equations(ode=ode)

# Print model information
print(model)

# Part 2
problem = OptimalControlProblem(model, x_0=[1], t_f=10, obj={"Q": 1, "R": 1})

# Part 3
# Initialize a DirectMethod to solve the OCP using collocation
solution_method = DirectMethod(
    problem, finite_elements=20, discretization_scheme="collocation"
)

# Solve the problem and get the result
result = solution_method.solve()

# Make one plot with the element x[0] (the first state) and one plot with the control u[0]
result.plot([{"x": [0]}, {"u": [0]}])
