from casadi import diag

from yaocptool.methods import DirectMethod, AugmentedLagrangian
from yaocptool.modelling import SystemModel, OptimalControlProblem


# Create Model
model = SystemModel(n_x=2, n_y=1, n_u=1, n_theta=1)
x_1 = model.x_sym[0]
x_2 = model.x_sym[1]

u = model.u_sym[0]
u_d = model.y_sym[0]

theta_activation = model.theta_sym[0]

a1 = 1
a2 = 1.1
ode = [-a1 * x_1 + u,
       -a2 * x_2 + u_d]
alg = [theta_activation * (u - u_d)]

model.include_system_equations(ode, alg)

# Create OCP
L = x_1 ** 2 + x_2 ** 2 + u ** 2 + u_d ** 2
problem = OptimalControlProblem(model, x_0=[1, 1], L=L, t_f=10.)

# Create solver
solution_method = AugmentedLagrangian(problem, DirectMethod,
                                      finite_elements=20,
                                      degree_control=4,
                                      degree=4,
                                      solver_options={'discretization_scheme': 'collocation'})
# Create the activation theta
theta = solution_method.create_constant_theta(constant=0, dimension=1)
theta[0] = 1
print(theta)

result = solution_method.solve(theta=theta)

result.plot([{'x': [0, 1]}, {'u': [0, 1]}])
