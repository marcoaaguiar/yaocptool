from casadi import DM, sqrt

from yaocptool.methods import DirectMethod
from yaocptool.modelling import SystemModel, StochasticOCP
from yaocptool.stochastic.pce import PCEConverter

######################
#  Main Variables    #
######################
# initial state and control

x_0 = DM([1, 1])
initial_control = [0.01]

# Prediction window, finite elements, and sampling time
prediction_window = 20
finite_elements = 20
t_s = prediction_window / finite_elements

######################
#      Model         #
######################
# Create a new model with 2 Tanks, the output of the first tank is connected on the second tank

model = SystemModel(name='2-Tanks')

# Get the symbolic variables
h_1, h_2 = model.create_state('h_1'), model.create_state('h_2')
u = model.create_control('u')

# Model Parameters
a = model.create_parameter('a')  # (m^2) Holes cross section
a_mean = model.create_parameter('a_mean')  # (m^2) Holes cross section
A = 28e-3  # (m^2) the tank area
g = 9.8  # gravitational acceleration

# Define the model ODEs
ode = [(u - a * sqrt(2 * g * h_1)) / A,
       (a * sqrt(2 * g * h_1) - a * sqrt(2 * g * h_2)) / A]

# Include the equations in the model
model.include_system_equations(ode=ode)

# Print the model for debugging purposes
print(model)

######################
#   StochasticOCP    #
######################
# Create the optimal control problem
problem = StochasticOCP(model)

# Define the problem initial conditions and final time
problem.x_0 = x_0
problem.u_guess = initial_control
problem.t_f = prediction_window

# Define the integrative cost ( cost = V(..., t_f) + \int_{t_0}^{t_f} L(...) dt) + \sum_{k} S(..., t_k)
problem.L = (h_1 - 2) ** 2 + 0 * (h_2 - 1) ** 2 + u ** 2

# We can change the bounds, e.g.: u_min, x_max, y_min, ...
problem.u_min = 0.00001

# include uncertain parameter
problem.set_parameter_as_uncertain_parameter(a, mean=a_mean, var=[1e-8], distribution='normal')

# Include chance constraint
problem.include_time_chance_inequality(a * sqrt(2 * g * h_1), rhs=[0.01], prob=[0.9])

######################
#      PCE           #
######################
# Apply PCE to transform the Stochastic OCP into a 'standard' OCP
pce_converter = PCEConverter(problem)

# It is possible to include some expressions if we want to calculate the statistics (mean and variance) with PCE
# pce_converter.stochastic_variables.append(a * sqrt(2 * g * h_1))
# pce_converter.stochastic_variables.append(h_1)

# Transform the Stochastic OCP into a OCP
pce_problem = pce_converter.convert_socp_to_ocp_with_pce()

######################
#  Solution Method   #
######################
# Initialize a DirectMethod to solve the OCP using collocation
solution_method = DirectMethod(pce_problem, finite_elements=20, discretization_scheme='collocation')

# Solve the problem and get the result
result = solution_method.solve(p=[0.071e-2] + [0] * 6)

# Make one plot with the element with h_0 and h_1 for every sample, notice that it has an additional state that is the
# cost at each sample.
result.plot(
    [{'x': range(0, 3 * pce_converter.n_samples, 3)},
     {'x': range(1, 3 * pce_converter.n_samples, 3)},
     {'u': [0]}]
)
