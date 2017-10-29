from yaocptool import create_constant_theta, join_thetas
from yaocptool.methods import DirectMethod, AugmentedLagrangian
from yaocptool.modelling import SystemModel, OptimalControlProblem

# Fixed parameters

feed = 1  # m^3/s

# Initialization
first_model = SystemModel(n_x=0, n_u=3)
models = []
for s in range(8):  # iterate for each scenario
    model = SystemModel(name='scenario_' + repr(s), n_x=6, n_y=9, n_u=0, n_theta=4)

    x_1 = model.x_sym[0]
    x_2 = model.x_sym[1]
    x_3 = model.x_sym[2]
    u_o1 = model.x_sym[3]
    u_o2 = model.x_sym[4]
    u_o3 = model.x_sym[5]

    u_f1 = model.y_sym[0]
    y_1 = model.y_sym[1]
    u_f2 = model.y_sym[2]
    y_2 = model.y_sym[3]
    u_f3 = model.y_sym[4]
    y_3 = model.y_sym[5]

    du_o1 = model.y_sym[6]
    du_o2 = model.y_sym[7]
    du_o3 = model.y_sym[8]

    d_1 = model.theta_sym[0]
    d_2 = model.theta_sym[1]
    d_3 = model.theta_sym[2]

    activation = model.theta_sym[3]

    ode = [u_f1 + d_1 - u_o1,
           u_f2 + d_2 - u_o2,
           u_f3 + d_3 - u_o3,
           du_o1,
           du_o2,
           du_o3]

    alg = [u_f1 - feed,
           y_1 - u_o1,
           #
           u_f2 - y_1,
           y_2 - u_o2,
           #
           u_f3 - y_2,
           y_3 - u_o3,
           #
           activation * (u_o1 - first_model.u_sym[0]),
           activation * (u_o2 - first_model.u_sym[1]),
           activation * (u_o3 - first_model.u_sym[2])
           ]

    model.include_system_equations(ode, alg)
    models.append(model)

first_model.merge(models)
# Create OCP
L = sum([(m.x_sym[0] - 2) ** 2 + (m.x_sym[1] - 2) ** 2 + (m.x_sym[2] - 2) ** 2
         + 0.1 * m.y_sym[6] ** 2 + 0.1 * m.y_sym[7] ** 2 + 0.1 * m.y_sym[8] ** 2 for m in models])

problem = OptimalControlProblem(first_model, x_0=[1, 1, 1, 0.5, 0.5, 0.5] * 8, L=L, t_f=10.)

# Create solver
solution_method = AugmentedLagrangian(problem, DirectMethod,
                                      finite_elements=20,
                                      degree_control=1,
                                      degree=4,
                                      # solver_options={'discretization_scheme': 'collocation'}
                                      )

# Create the parameters variation and activation theta
m_thetas = []
for k, m in enumerate(models):
    m_theta = create_constant_theta(constant=[0.1, 0.1, 0.1, 0], dimension=4,
                                    finite_elements=solution_method.finite_elements)
    m_theta[0][-1] = 1
    for i in range(solution_method.finite_elements):
        if k % 2 == 1:
            m_theta[i][0] = 0.3
        if k % 4 >= 2:
            m_theta[i][1] = 0.3
        if k % 8 >= 2:
            m_theta[i][1] = 0.3
    m_thetas.append(m_theta)

theta = join_thetas(*m_thetas)
result = solution_method.solve(theta=theta)

result.plot([{'x': range(0, first_model.n_x, 6)}, {'x': range(1, first_model.n_x, 6)}, {'x': range(2, first_model.n_x, 6)},
             {'x': range(3, first_model.n_x, 6)}, {'x': range(4, first_model.n_x, 6)}, {'x': range(5, first_model.n_x, 6)}
             ])
