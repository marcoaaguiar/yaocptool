from yaocptool.modelling import SystemModel
from casadi import sqrt, DM

# Create a new model with 2 Tanks, the output of the first tank is connected on the second tank
model = SystemModel(n_x=2, n_u=1, name='2-Tanks')

# Get the symbolic variables
h_1, h_2 = model.x_sym[0], model.x_sym[1]
u = model.u_sym

# Model Parameters
A = 28e-4  # (m^2) the tank area
a = 0.071e-4  # (m^2) Holes cross section
g = 9.8  # gravitational acceleration

# Define the model ODEs
ode = [(u - a * sqrt(2 * g * h_1)) / A,
       (a * sqrt(2 * g * h_1) - a * sqrt(2 * g * h_2)) / A]

# Include the equations in the model
model.include_system_equations(ode=ode)

# Just check if the where correctly included
print(model)

# Do a simple simulation
x_0 = DM([0.101211, 0.101211])
sim_result = model.simulate(x_0=x_0, t_f=[i*0.2 for i in range(1, 201)], p=[1.1 * 1e-5])

# sim_result is a SimulationResult object
# To plot the results of the simulation use the .plot function
# It takes as input a list of dictionaries, each dictionary represents a plot
# In the dictionary use keyword 'x' to specify which states you want to print
# The value should be a list of state to be printed.
# The keywords that are acceptable are: 'x', 'y', 'u'
sim_result.plot([{'x': [0]}])

# Linearize the model at a given point
linearized_model = model.linearize(x_bar=x_0, u_bar=[1e-5])
x_f2 = linearized_model.simulate(x_0=[0, 0], t_f=100., p=[0.1 * 1e-5])

# print(x_0 + x_f2['xf'])
