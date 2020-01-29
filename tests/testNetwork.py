from casadi import sqrt, pi, sum1, fmax

from yaocptool import find_variables_indices_in_vector
from yaocptool.methods import DirectMethod, AugmentedLagrangian
from yaocptool.methods.network.distributedaugmetedlagrangian import DistributedAugmentedLagrangian
from yaocptool.modelling import SystemModel, OptimalControlProblem, Network

t_f = 200
dal_options = {'finite_elements': 60,
               'degree': 3,
               'mu_0': 1e2,
               'mu_max': 1e20,
               'abs_tol': 1e-6,
               'max_iter_inner': 5,
               'max_iter_inner_first': None,
               'max_iter_outer': 10}


class Tank(SystemModel):
    def __init__(self, tank_id, n_in=None, **kwargs):
        SystemModel.__init__(self, model_name_as_prefix=True, **kwargs)
        if n_in is None:
            n_in = 2 if tank_id <= 2 else 1

        vol = self.create_state('vol')
        w_out = self.create_algebraic_variable('w_out')
        w_in = self.create_input('w_in', n_in)

        g = 9.8
        big_a = pi * 0.1 ** 2
        a = pi * 0.01 ** 2
        self.include_system_equations(ode=[sum1(w_in) - w_out],
                                      alg=[w_out - a * sqrt(fmax(1e-12, 2 * g * vol / big_a))])
        # alg=[w_out - a * (2 * g * vol / big_a)])


class Pump(SystemModel):
    def __init__(self, pump_id, n_out=2, **kwargs):
        SystemModel.__init__(self, model_name_as_prefix=True, **kwargs)

        w_out = self.create_algebraic_variable('w_out', n_out)
        u = self.create_control('u')

        gamma = 0.3 if pump_id == 0 else 0.5
        k = 5.6e-4  # [m³/V]

        flow = k * u
        if n_out == 2:
            self.include_system_equations(alg=[w_out[0] - gamma * flow,
                                               w_out[1] - (1 - gamma) * flow])
        else:
            self.include_system_equations(alg=[w_out[0] - flow])


class Pump2(SystemModel):
    def __init__(self, pump_id, n_out=2, **kwargs):
        SystemModel.__init__(self, model_name_as_prefix=True, **kwargs)

        u = self.create_state('u')
        w_out = self.create_algebraic_variable('w_out', n_out)
        du = self.create_control('du')

        gamma = 0.3 if pump_id == 1 else 0.5
        k = 5.6e-4  # [m³/V]

        flow = k * u
        self.include_system_equations(ode=[du])
        if n_out == 2:
            self.include_system_equations(alg=[w_out[0] - gamma * flow,
                                               w_out[1] - (1 - gamma) * flow])
        else:
            self.include_system_equations(alg=[w_out[0] - flow])


def create_four_tanks():
    # Create the Network
    n = Network(name='Four_Tanks')
    nodes_tank = []
    nodes_pumps = []

    # Create tank nodes
    for i in range(4):
        ind = i + 1
        tank_model = Tank(tank_id=ind, name='Tank_' + str(ind))
        tank_problem = OptimalControlProblem(name='OCP_tank_' + str(ind), model=tank_model, x_0=[0.2], t_f=t_f)
        tank_problem.L = (tank_model.x - (0.4 + i * 0.2)) ** 2 if ind <= 2 else 0
        tank_problem.u_guess = [0.0025 * (i + 1) for i in range(tank_model.n_u)]
        tank_problem.y_guess = [0.005] * tank_model.n_y
        tank_problem.x_min = 0.01
        tank_problem.y_min = [0] * tank_model.n_y
        node = n.create_node(name='tank_' + str(ind), model=tank_model, problem=tank_problem)
        nodes_tank.append(node)

    # Create pump nodes
    for i in range(2):
        ind = i + 1
        pump_model = Pump2(pump_id=ind, name='Pump_' + str(ind))
        pump_problem = OptimalControlProblem(name='OCP_pump_' + str(ind), model=pump_model,
                                             x_0=[4] * pump_model.n_x, t_f=t_f)
        pump_problem.L = 1e-2 * pump_model.u ** 2
        pump_problem.y_guess = [0.0025 for i in range(pump_model.n_y)]
        pump_problem.u_guess = [0.25 for i in range(pump_model.n_u)]
        pump_problem.y_min = [0 for i in range(pump_model.n_y)]

        # pump_problem.u_max = 5
        # pump_problem.u_min = -5

        node = n.create_node(name='pump_' + str(ind), model=pump_model, problem=pump_problem)
        nodes_pumps.append(node)

    # Pump Connections
    n.connect(nodes_pumps[0].model.y[0], nodes_tank[0].model.u[0], nodes_pumps[0], nodes_tank[0])
    n.connect(nodes_pumps[0].model.y[1], nodes_tank[3].model.u[0], nodes_pumps[0], nodes_tank[3])
    n.connect(nodes_pumps[1].model.y[0], nodes_tank[1].model.u[0], nodes_pumps[1], nodes_tank[1])
    n.connect(nodes_pumps[1].model.y[1], nodes_tank[2].model.u[0], nodes_pumps[1], nodes_tank[2])

    # Tank connections
    n.connect(nodes_tank[2].model.y, nodes_tank[0].model.u[1], nodes_tank[2], nodes_tank[0])
    n.connect(nodes_tank[3].model.y, nodes_tank[1].model.u[1], nodes_tank[3], nodes_tank[1])

    return n


def create_pump_tank():
    n = Network(name='pump_tank')

    # create pump
    ind = 1
    pump_model = Pump2(pump_id=ind, name='Pump_' + str(ind), n_out=1)
    pump_problem = OptimalControlProblem(name='OCP_pump_' + str(ind), model=pump_model, x_0=[1] * pump_model.n_x,
                                         t_f=t_f)
    pump_problem.L = 1e-0 * (pump_model.u) ** 2
    pump_problem.u_max = 5
    pump_problem.u_min = -5

    pump_node = n.create_node(name='pump_' + str(ind), model=pump_model, problem=pump_problem)

    # create tank
    tank_model = Tank(tank_id=ind, name='Tank_' + str(ind), n_in=1)
    tank_problem = OptimalControlProblem(name='OCP_tank_' + str(ind), model=tank_model, x_0=[0.2], t_f=t_f)
    tank_problem.L = (tank_model.x - 0.15) ** 2
    tank_problem.x_min = 0.05
    tank_problem.u_guess = [0.3 * (i + 1) for i in range(tank_model.n_u)]
    tank_problem.y_guess = [0.2] * tank_model.n_y
    # tank_problem.x_min = 0.01
    tank_problem.y_min = [0] * tank_model.n_y
    tank_node = n.create_node(name='tank_' + str(ind), model=tank_model, problem=tank_problem)

    n.connect(pump_model.y[0], tank_model.u[0], pump_node, tank_node)
    # n.connect(pump_model.y[1], tank_model.u[1], pump_node, tank_node)

    return n


def create_two_pumps_two_tanks():
    # Create the Network
    n = Network(name='two_pumps_two_tanks')
    nodes_tank = []
    nodes_pumps = []

    # Create tank nodes
    for i in range(2):
        ind = i + 1
        tank_model = Tank(tank_id=ind, name='Tank_' + str(ind))
        tank_problem = OptimalControlProblem(name='OCP_tank_' + str(ind), model=tank_model, x_0=[0.2], t_f=t_f)
        tank_problem.L = (tank_model.x - (0.4 + i * 0.2)) ** 2 if ind <= 2 else 0
        tank_problem.u_guess = [0.0025 * (i + 1) for i in range(tank_model.n_u)]
        tank_problem.y_guess = [0.005] * tank_model.n_y
        tank_problem.x_min = 0.01
        tank_problem.y_min = [0] * tank_model.n_y
        node = n.create_node(name='tank_' + str(ind), model=tank_model, problem=tank_problem)
        nodes_tank.append(node)

    # Create pump nodes
    for i in range(2):
        ind = i + 1
        pump_model = Pump2(pump_id=ind, n_out=2, name='Pump_' + str(ind))
        pump_problem = OptimalControlProblem(name='OCP_pump_' + str(ind), model=pump_model,
                                             x_0=[4] * pump_model.n_x, t_f=t_f)
        pump_problem.L = 1e-2 * pump_model.u ** 2
        pump_problem.y_guess = [0.0025 for i in range(pump_model.n_y)]
        pump_problem.u_guess = [0.25 for i in range(pump_model.n_u)]
        pump_problem.y_min = [0 for i in range(pump_model.n_y)]

        # pump_problem.u_max = 5
        # pump_problem.u_min = -5

        node = n.create_node(name='pump_' + str(ind), model=pump_model, problem=pump_problem)
        nodes_pumps.append(node)

    # Pump Connections
    n.connect(nodes_pumps[0].model.y[0], nodes_tank[0].model.u[0], nodes_pumps[0], nodes_tank[0])
    n.connect(nodes_pumps[0].model.y[1], nodes_tank[1].model.u[0], nodes_pumps[0], nodes_tank[1])
    n.connect(nodes_pumps[1].model.y[0], nodes_tank[0].model.u[1], nodes_pumps[1], nodes_tank[0])
    n.connect(nodes_pumps[1].model.y[1], nodes_tank[1].model.u[1], nodes_pumps[1], nodes_tank[1])

    # Tank connections
    # n.connect(nodes_tank[2].model.y, nodes_tank[0].model.u[1], nodes_tank[2], nodes_tank[0])
    # n.connect(nodes_tank[3].model.y, nodes_tank[1].model.u[1], nodes_tank[3], nodes_tank[1])

    return n


n = create_four_tanks()
# n = create_pump_tank()
# n = create_two_pumps_two_tanks()
# Draw the networkTrue
# n.plot()
n.insert_intermediary_nodes()

CLASSIC = False
CENTRALIZED_AUG_LAG = False
DISTRIBUTED = True

if CLASSIC:
    centralized_problem = n.get_problem()
    c_solution_method = DirectMethod(centralized_problem,
                                     discretization_scheme='collocation',
                                     initial_guess_heuristic='problem_info',
                                     finite_elements=40,
                                     degree=3,
                                     degree_control=3)
    rc = c_solution_method.solve()
    rc.plot([{'x': ['Tank_._']}, {'x': ['Pump_']}, {'u': 'all'}])

if CENTRALIZED_AUG_LAG:
    centralized_problem = n.get_problem()
    relaxed_algebraics = []
    relax_algebraic_var_index = []
    for i, c in enumerate(n.connections):
        relaxed_algebraics.extend(find_variables_indices_in_vector(n.connections[c]['u'] - n.connections[c]['y'],
                                                                   centralized_problem.model.alg,
                                                                   depth=10))
        relax_algebraic_var_index.extend(find_variables_indices_in_vector(n.connections[c]['u'],
                                                                          centralized_problem.model.y,
                                                                          depth=10))

    c_solution_method = AugmentedLagrangian(centralized_problem,
                                            ocp_solver_class=DirectMethod,
                                            discretization_scheme='collocation',
                                            initial_guess_heuristic='problem_info',
                                            finite_elements=60,
                                            degree=3,
                                            degree_control=3,
                                            relax_algebraic_index=relaxed_algebraics,
                                            relax_algebraic_var_index=relax_algebraic_var_index
                                            )
    ral = c_solution_method.solve()
    ral.plot([{'x': ['Tank_._']}, {'x': ['Pump_']}, {'u': ['Tank_._w_in', 'Pump_._w_out']}])

if DISTRIBUTED:
    solution_method = DistributedAugmentedLagrangian(network=n,
                                                     solution_method_class=DirectMethod,
                                                     solution_method_options={'discretization_scheme': 'collocation',
                                                                              # 'initial_guess_heuristic': 'problem_info'
                                                                              },
                                                     **dal_options)

    r = solution_method.solve()
    #
    # for node in r:
    #     r[node].plot([{'x': ['Tank_._']}, {'x': ['Pump_']}, {'u': 'all'}])
    #
    solution_method.plot_all_relaxations(r)

# r_pump = r[n.get_node_by_id(0)]
# r_tank = r[n.get_node_by_id(1)]
#
# r_pump.plot([{'y': [0]}, {'u': [0], 'x': [0]}, {'x': [-1]}, {}])
# r_tank.plot([{'u': [0], 'y': [0]}, {}, {'x': [-1]}, {'x': [0]}])
