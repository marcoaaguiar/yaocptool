from yaocptool.estimation.unscented_kalman_filter import UnscentedKalmanFilter
from yaocptool.methods import DirectMethod
from yaocptool.modelling import SystemModel, OptimalControlProblem
from casadi import sqrt, DM, diag
from yaocptool.mpc import PlantSimulation, MPC

######################
#  Main Variables    #
######################
# Measurement matrix
c_matrix = DM([[0, 1]])

# initial state and control
x_0 = DM([1, 1])
initial_control = [0.01]

# Prediction window, finite elements, and sampling time
prediction_window = 20.
finite_elements = 20
t_s = prediction_window/finite_elements

######################
#      Model         #
######################
# Create a new model with 2 Tanks, the output of the first tank is connected on the second tank
model = SystemModel(name='2-Tanks')

# Get the symbolic variables
h_1, h_2 = model.create_state('h_1'), model.create_state('h_2')
u = model.create_control('u')

# Model Parameters
A = 28e-3  # (m^2) the tank area
a = 0.071e-2  # (m^2) Holes cross section
g = 9.8  # gravitational acceleration

# Define the model ODEs
ode = [(u - a * sqrt(2 * g * h_1)) / A,
       (a * sqrt(2 * g * h_1) - a * sqrt(2 * g * h_2)) / A]

# Include the equations in the model
model.include_system_equations(ode=ode)

# Print the model for debugging purposes
print(model)

######################
#      OCP           #
######################
# Create the optimal control problem
problem = OptimalControlProblem(model)

# Define the problem initial conditions and final time
problem.x_0 = x_0
problem.u_guess = initial_control
problem.t_f = prediction_window

# Define the integrative cost ( cost = V(..., t_f) + \int_{t_0}^{t_f} L(...) dt) + \sum_{k} S(..., t_k)
problem.L = (h_1 - 2) ** 2 + 0 * (h_2 - 1) ** 2 + u ** 2

# We can change the bounds, e.g.: u_min, x_max, y_min, ...
problem.u_min = 0.001
problem.x_min = [0.05, 0.05]

# Or include other constraints (g(...) <= 0)
problem.include_time_inequality(h_1 + h_2 - 10)

######################
#  Solution Method   #
######################
# Solution method will solve the OCP every iteration
solution_method = DirectMethod(problem,
                               finite_elements=20,
                               discretization_scheme='collocation',
                               )

######################
#   Estimator       #
######################
# Create a copy of the model for the estimator
estimator_model = model.get_copy()

# Create the estimator
estimator = UnscentedKalmanFilter(model=estimator_model, t_s=t_s,
                                  p_k=0.0001 * DM.eye(estimator_model.n_x),
                                  x_mean=x_0 * 1.1, c_matrix=c_matrix,
                                  r_v=0.00001 * DM.eye(estimator_model.n_x),
                                  r_n=0.00001 * DM.eye(1),
                                  )

######################
#      Plant         #
######################
# Create a copy of the model for the plant
plant_model = model.get_copy()

# create the plant
plant = PlantSimulation(model=plant_model, x_0=x_0, c_matrix=c_matrix, u=0.1)

######################
#        MPC         #
######################
# MPC
mpc = MPC(plant, solution_method, estimator=estimator, include_cost_in_state_vector=True)

# Run the plant without estimator or control
mpc.run_fixed_control(initial_control, 20)

# Run only with the estimator applying a fixed control
mpc.run_fixed_control_with_estimator(initial_control, 20)

# Run the plant with estimator and calculating the control
mpc.run(100)

mpc.plant.simulation_results.plot([{'x': 'all'}, {'u': 'all'}])
