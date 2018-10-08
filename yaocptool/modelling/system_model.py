# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 10:50:48 2016

@author: marco
"""
import collections
import copy
from casadi import SX, vertcat, substitute, Function, jacobian, mtimes, rootfinder, vec, horzcat

from yaocptool import remove_variables_from_vector, config
from yaocptool.modelling import DAESystem, SimulationResult


# TODO: Check linearize method


class SystemModel:
    def __init__(self, name='model', n_x=0, n_y=0, n_z=0, n_u=0, n_p=0, n_theta=0, **kwargs):
        """
            x - states
            y - (internal) algebraic
            z - external algebraic
            u - control
            p - constant parameters
            theta - parameters dependent of the finite_elements
            u_par - parametrized control parameters

            Note: when vectorizing the parameters order is [ p; theta; u_par]
        """
        self.name = name

        self.ode = vertcat([])  # ODE
        self.alg = vertcat([])  # Algebraic equations

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if 'x_names' in kwargs and n_x == 0:
            self.x_sym = []
            for name_tuple in kwargs['x_names']:
                name = name_tuple[0]
                size = name_tuple[1]
                self.x_sym = vertcat(self.x_sym, SX.sym(self.name + '_' + name, size))
        else:
            self.x_sym = SX.sym(self.name + '_x', n_x)
            # raise Exception("You should provide an 'n_x' OR a 'x_names' dict.")

        self.x_0_sym = SX.sym(self.name + '_x_0_sym', self.n_x)
        self.y_sym = SX.sym(self.name + '_y', n_y)
        self.u_sym = SX.sym(self.name + '_u', n_u)
        self.p_sym = SX.sym(self.name + '_p', n_p)
        self.theta_sym = SX.sym(self.name + '_theta', n_theta)

        self.t_sym = SX.sym('t')
        self.tau_sym = SX.sym('tau')

        self.u_par = vertcat(self.u_sym)
        self.u_func = vertcat(self.u_sym)

        self.hasAdjointVariables = False

    @property
    def system_type(self):
        if self.n_y > 0:
            return 'dae'
        else:
            return 'ode'

    @property
    def n_x(self):
        return self.x_sym.numel()

    @property
    def n_y(self):
        return self.y_sym.numel()

    @property
    def n_u(self):
        return self.u_sym.numel()

    @property
    def n_p(self):
        return self.p_sym.numel()

    @property
    def n_theta(self):
        return self.theta_sym.numel()

    @property
    def x_sys_sym(self):
        if self.hasAdjointVariables:
            return self.x_sym[:self.n_x / 2]
        else:
            return self.x_sym

    @property
    def lamb_sym(self):
        if self.hasAdjointVariables:
            return self.x_sym[self.n_x / 2:]
        else:
            return SX()

    @property
    def all_sym(self):
        return self.t_sym, self.x_sym, self.y_sym, self.u_sym, self.p_sym, self.theta_sym, self.u_par

    @property
    def x(self):
        return self.x_sym

    @x.setter
    def x(self, value):
        self.x_sym = value

    @property
    def y(self):
        return self.y_sym

    @y.setter
    def y(self, value):
        self.y_sym = value

    @property
    def u(self):
        return self.u_sym

    @u.setter
    def u(self, value):
        self.u_sym = value

    @property
    def p(self):
        return self.p_sym

    @p.setter
    def p(self, value):
        self.p_sym = value

    @property
    def theta(self):
        return self.theta_sym

    @theta.setter
    def theta(self, value):
        self.theta_sym = value

    @property
    def x_names(self):
        return [self.x_sym[i].name() for i in range(self.n_x)]

    @property
    def y_names(self):
        return [self.y_sym[i].name() for i in range(self.n_y)]

    @property
    def u_names(self):
        return [self.u_sym[i].name() for i in range(self.n_u)]

    @property
    def p_names(self):
        return [self.p_sym[i].name() for i in range(self.n_p)]

    @property
    def theta_names(self):
        return [self.theta_sym[i].name() for i in range(self.n_theta)]

    def __repr__(self):
        s = ''
        s += '=' * 20 + '\n'
        s += 'Model Name: {:>23}'.format(self.name)
        s += '| System type:                            {:>3}'.format(self.system_type)
        s += '\n'
        s += '-' * 20 + '\n'
        s += 'Number of states (x):         {:4} | Number of algebraic (y):               {:4}'.format(self.n_x,
                                                                                                       self.n_y)
        s += '\n'
        s += 'Number of controls (u):       {:4} |'.format(self.n_u)
        s += '\n'
        s += 'Number of parameters (p):     {:4} | Number of finite elem. param. (theta): {:4}'.format(self.n_p,
                                                                                                       self.n_theta)
        s += '\n'
        s += '-' * 20 + '\n'
        s += 'Number of ODE:                {:4} | Number of algebraic eq.:               {:4}'.format(
            self.ode.numel(),
            self.alg.numel())
        s += '\n'
        s += '=' * 20 + '\n'
        return s

    def replace_variable(self, original, replacement):
        """
            Replace a variable or parameter by an variable or expression.
            :param replacement:
            :param original: SX: and replacement, and also variable type which
            describes which type of variable is being remove to it from the
            counters. Types: 'x', 'y', 'u', 'p', 'ignore'
        """
        original = vertcat(original)
        replacement = vertcat(replacement)
        if not original.numel() == replacement.numel():
            raise ValueError("Original and replacement must have the same number of elements!"
                             "original.numel()={}, replacement.numel()={}".format(original.numel(),
                                                                                  replacement.numel()))

        if original.numel() > 0:
            print('Replacing: {} with {}'.format(original, replacement))
            self.ode = substitute(self.ode, original, replacement)
            self.alg = substitute(self.alg, original, replacement)

            #TODO: Im commenting the  following line because I think they are wrong
            # self.u_par = substitute(self.u_par, original, replacement)
            self.u_func = substitute(self.u_func, original, replacement)

    def create_state(self, name='x', size=1):
        """
        Create a new state with the name "name" and size "size".
        Size can be an int or a tuple (e.g. (2,2)). However, the new state will be vectorized (casadi.vec) to be
        included in the state vector (model.x_sym).
        :param name: str
        :param size: int|tuple
        :return:
        """
        new_x = SX.sym(name, size)
        new_x_0_sym = SX.sym(name + '_0_sym', size)
        self.include_state(vec(new_x), ode=None, x_0_sym=vec(new_x_0_sym))
        return new_x

    def create_algebraic_variable(self, name='y', size=1):
        """
        Create a new algebraic variable with the name "name" and size "size".
        Size can be an int or a tuple (e.g. (2,2)). However, the new algebraic variable will be vectorized (casadi.vec)
        to be included in the algebraic vector (model.y_sym).
        :param name: str
        :param size: int or tuple
        :return:
        """
        new_y = SX.sym(name, size)
        self.include_algebraic(vec(new_y))
        return new_y

    def create_control(self, name='u', size=1):
        """
        Create a new control variable name "name" and size "size".
        Size can be an int or a tuple (e.g. (2,2)). However, the new control variable will be vectorized (casadi.vec)
        to be included in the control vector (model.u_sym).
        :param name: str
        :param size: int
        :return:
        """
        new_u = SX.sym(name, size)
        self.include_control(vec(new_u))
        return new_u

    def create_parameter(self, name='p', size=1):
        """
        Create a new parameter name "name" and size "size"
        :param name: str
        :param size: int
        :return:
        """
        new_p = SX.sym(name, size)
        self.include_parameter(vec(new_p))
        return new_p

    def create_theta(self, name='theta', size=1):
        """
        Create a new parameter name "name" and size "size"
        :param name: str
        :param size: int
        :return:
        """
        new_theta = SX.sym(name, size)
        self.include_theta(vec(new_theta))
        return new_theta

    # region INCLUDES

    def include_system_equations(self, ode=None, alg=None):
        if ode is not None:
            if isinstance(ode, list):  # if ode is list or tuple
                ode = vertcat(*ode)
            self.ode = vertcat(self.ode, ode)

        if alg is not None:
            if isinstance(alg, list):  # if alg is list or tuple
                alg = vertcat(*alg)
            self.alg = vertcat(self.alg, alg)

    def include_state(self, var, ode=None, x_0_sym=None):
        if ode is None:
            ode = []
        delta_n_x = var.numel()
        self.x_sym = vertcat(self.x_sym, var)
        if x_0_sym is None:
            x_0_sym = SX.sym('x_0_sym', delta_n_x)
        self.x_0_sym = vertcat(self.x_0_sym, x_0_sym)

        self.include_system_equations(ode=ode)
        return x_0_sym

    def include_algebraic(self, var, alg=None):
        self.y_sym = vertcat(self.y_sym, var)

        self.include_system_equations(alg=alg)

    def include_control(self, var):
        self.u_sym = vertcat(self.u_sym, var)
        self.u_par = vertcat(self.u_par, var)

    def include_parameter(self, p):
        self.p_sym = vertcat(self.p_sym, p)

    def include_theta(self, theta):
        self.theta_sym = vertcat(self.theta_sym, theta)

    # endregion

    # region REMOVE

    def remove_algebraic(self, var, eq=None):
        self.y_sym = remove_variables_from_vector(var, self.y_sym)
        if eq is not None:
            self.alg = remove_variables_from_vector(eq, self.alg)

    def remove_control(self, var):
        self.u_sym = remove_variables_from_vector(var, self.u_sym)

    def remove_parameter(self, var):
        self.p_sym = remove_variables_from_vector(var, self.p_sym)

    def remove_theta(self, var):
        self.theta_sym = remove_variables_from_vector(var, self.theta_sym)

    # endregion
    # ==============================================================================
    # region # Standard Function Call
    # ==============================================================================

    @staticmethod
    def put_values_in_all_sym_format(t=None, x=None, y=None, u=None, p=None, theta=None, u_par=None):
        if t is None:
            t = []
        if x is None:
            x = []
        if y is None:
            y = []
        if u is None:
            u = []
        if p is None:
            p = []
        if theta is None:
            theta = []
        if u_par is None:
            u_par = []
        return t, x, y, u, p, theta, u_par

    # endregion
    # ==============================================================================

    # ==============================================================================
    # region # TIME
    # ==============================================================================

    def convert_expr_from_time_to_tau(self, expr, t_k, t_kp1):
        t = self.t_sym
        tau = self.tau_sym
        h = t_kp1 - t_k
        return substitute(expr, t, tau * h + t_k)

    def convert_expr_from_tau_to_time(self, expr, t_k, t_kp1):
        t = self.t_sym
        tau = self.tau_sym
        h = t_kp1 - t_k
        return substitute(expr, tau, (t - t_k) / h)

    # endregion
    # ==============================================================================

    # ==============================================================================
    # region MERGE
    # ==============================================================================

    def merge(self, models_list, connecting_equations=None):
        if not isinstance(models_list, list):
            models_list = [models_list]

        for model in models_list:
            model.replace_variable(model.t_sym, self.t_sym)
            model.replace_variable(model.tau_sym, self.tau_sym)
            self.include_state(model.x_sym, model.ode, model.x_0_sym)
            self.include_algebraic(model.y_sym, model.alg)
            self.include_control(model.u_sym)
            self.include_parameter(model.p_sym)
            self.include_theta(model.theta_sym)

        self.include_system_equations(alg=connecting_equations)

    def get_copy(self):
        """
            Get a copy of this model
        :rtype: SystemModel
        """
        return copy.copy(self)

    def get_hardcopy(self):
        """
            Get a hard copy of this model, differently from "get_copy", the variables of the original copy and the
            hard copy will not be the same, i.e. model.x_sym != copy.x_sym

        :rtype: SystemModel
        """
        model_copy = SystemModel(name=self.name)
        x_copy = vertcat([])
        y_copy = vertcat([])
        u_copy = vertcat([])
        p_copy = vertcat([])
        theta_copy = vertcat([])

        if self.n_x > 0:
            x_copy = vertcat(*[model_copy.create_state(self.x_sym[i].name()) for i in range(self.n_x)])
        if self.n_y > 0:
            y_copy = vertcat(*[model_copy.create_algebraic_variable(self.y_sym[i].name()) for i in range(self.n_y)])
        if self.n_u > 0:
            u_copy = vertcat(*[model_copy.create_control(self.u_sym[i].name()) for i in range(self.n_u)])

        if self.n_p > 0:
            p_copy = vertcat(*[model_copy.create_parameter(self.p_sym[i].name()) for i in range(self.n_p)])
        if self.n_theta > 0:
            theta_copy = vertcat(*[model_copy.create_theta(self.theta_sym[i].name()) for i in range(self.n_theta)])

        model_copy.include_system_equations(ode=self.ode, alg=self.alg)
        model_copy.u_func = self.u_func
        model_copy.u_par = self.u_par

        model_copy.replace_variable(self.x_sym, x_copy)
        model_copy.replace_variable(self.y_sym, y_copy)
        model_copy.replace_variable(self.u_sym, u_copy)
        model_copy.replace_variable(self.p_sym, p_copy)
        model_copy.replace_variable(self.theta_sym, theta_copy)

        model_copy.hasAdjointVariables = self.hasAdjointVariables

        return model_copy

    # endregion
    # ==============================================================================

    # ==============================================================================
    # region SIMULATION
    # ==============================================================================

    def get_dae_system(self):
        """ Return a DAESystem object with the model equations.

        :return: DAESystem
        """
        if self.system_type == 'ode':
            kwargs = {'x': self.x_sym, 'ode': self.ode, 't': self.t_sym, 'tau': self.tau_sym}
        else:
            kwargs = {'x': self.x_sym, 'y': self.y_sym, 'ode': self.ode,
                      'alg': self.alg, 't': self.t_sym, 'tau': self.tau_sym}
        if self.n_p + self.n_theta + self.u_par.numel() > 0:
            kwargs['p'] = vertcat(self.p_sym, self.theta_sym, self.u_par)

        return DAESystem(**kwargs)

    def simulate(self, x_0, t_f, t_0=0.0, u=None, p=None, theta=None, y_0=None, integrator_type='implicit',
                 integrator_options=None):
        """ Simulate model.
            If t_f is a float, then only one simulation will be done. If t_f is a list of times, then a sequence of
            simulations will be done, that each t_f is the end of a finite element.

        :param list||DM x_0: Initial condition
        :param float||list t_f: Final time of the simulation, can be a list of final times for sequential simulation
        :param float t_0: Initial time
        :param list||DM u: Controls of the system to be simulated
        :param DM||SX||list p: Simulation parameters
        :param dict theta: Parameters theta, which varies for each simulation for sequential simulations.
                           If t_f is a list then theta has to have one entry for each k in [0,...,len(t_f)]
        :param y_0: Initial guess for the algebraic variables
        :param str integrator_type: 'implicit' or 'explicit'
        :param dict integrator_options: options to be passed to the integrator

        :rtype: SimulationResult
        """

        if isinstance(x_0, collections.Iterable):
            x_0 = vertcat(x_0)
        if not isinstance(t_f, collections.Iterable):
            t_f = [t_f]

        if theta is None:
            theta = dict([(k, []) for k in range(len(t_f))])
        if integrator_options is None:
            integrator_options = {}
        if p is None:
            p = []

        if u is None:  # if control is not given
            u = [[]] * len(t_f)
        elif not isinstance(u, (list, tuple)):  # if control is given as number or a casadi object
            u = [u] * len(t_f)

        if len(t_f) > 1 and not len(u) == len(t_f):
            raise ValueError('If "t_f" is a list, the parameter "u" should be a list with same length of "t_f"')

        dae_sys = self.get_dae_system()

        t_x_list = [t_0]
        t_yu_list = []
        x_list = [x_0]
        y_list = []
        u_list = []
        t_k = t_0
        x_k = x_0
        y_k = y_0
        for k, t_kpp in enumerate(t_f):
            p_k = vertcat(p, theta[k], u[k])
            result = dae_sys.simulate(x_0=x_k, t_f=t_kpp, t_0=t_k, p=p_k, y_0=y_k,
                                      integrator_type=integrator_type, integrator_options=integrator_options)
            t_x_list.append(t_kpp)
            t_yu_list.append(t_kpp)
            x_list.append(result['xf'])
            y_list.append(result['zf'])
            u_list.append(u[k])

            t_k = t_kpp
            x_k = result['xf']
            y_k = result['zf']
        t_x_list = horzcat(*t_x_list)
        t_yu_list = horzcat(*t_yu_list)

        simulation_result = SimulationResult(model_name=self.name,
                                             n_x=self.n_x, n_y=self.n_y, n_u=self.n_u,
                                             x_names=self.x_names, y_names=self.y_names, u_names=self.u_names,
                                             t_0=t_0, t_f=t_f[-1], finite_elements=len(t_f))

        simulation_result.insert_data('x', time=t_x_list, value=horzcat(*x_list))
        simulation_result.insert_data('y', time=t_yu_list, value=horzcat(*y_list))
        simulation_result.insert_data('u', time=t_yu_list, value=horzcat(*u_list))

        return simulation_result

    def simulate_raw(self, x_0, t_f, t_0=0.0, p=None, y_0=None, integrator_type='implicit',
                     integrator_options=None):
        """ Perform a single simulation.

        :param list||DM x_0: Initial condition
        :param float t_f: Final time of the simulation
        :param float t_0: Initial time
        :param DM||SX||list p: Simulation parameters
        :param y_0: Initial guess for the algebraic variables
        :param str integrator_type: 'implicit' or 'explicit'
        :param dict integrator_options: options to be passed to the integrator
        """

        if integrator_options is None:
            integrator_options = {}
        if p is None:
            p = []
        if isinstance(x_0, collections.Iterable):
            x_0 = vertcat(x_0)

        dae_sys = self.get_dae_system()

        result = dae_sys.simulate(x_0=x_0, t_f=t_f, t_0=t_0, p=p, y_0=y_0,
                                  integrator_type=integrator_type, integrator_options=integrator_options)
        return result

    # endregion

    def find_algebraic_variable(self, x, u, guess=None, t=0.0, p=None, theta_value=None, rootfinder_options=None):
        if guess is None:
            guess = [1] * self.n_y
        if rootfinder_options is None:
            rootfinder_options = dict(nlpsol='ipopt', nlpsol_options=config.SOLVER_OPTIONS['nlpsol_options'])
        if p is None:
            p = []
        if theta_value is None:
            theta_value = []

        # replace known variables
        alg = self.alg
        known_var = vertcat(self.t_sym, self.x_sym, self.u_sym, self.p_sym, self.theta_sym)
        known_var_values = vertcat(t, x, u, p, theta_value)
        alg = substitute(alg, known_var, known_var_values)

        f_alg = Function('f_alg', [self.y_sym], [alg])

        rf = rootfinder('rf_algebraic_variable', 'nlpsol', f_alg, rootfinder_options)
        res = rf(guess)
        return res

    def linearize(self, x_bar, u_bar):
        """
        Returns a linearized model at a given points (X_BAR, U_BAR)
        """
        a_matrix = Function('a_matrix', [self.x_sym, self.u_sym], [jacobian(self.ode, self.x_sym)])(x_bar, u_bar)
        b_matrix = Function('b_matrix', [self.x_sym, self.u_sym], [jacobian(self.ode, self.u_sym)])(x_bar, u_bar)

        linear_model = SystemModel(n_x=self.n_x, n_u=self.n_u)
        linear_model.include_system_equations(
            ode=[mtimes(a_matrix, linear_model.x_sym) + mtimes(b_matrix, linear_model.u_sym)])
        linear_model.name = 'linearized_' + self.name

        linear_model.a_matrix = a_matrix
        linear_model.b_matrix = b_matrix

        return linear_model

    def find_equilibrium(self, additional_eqs, guess=None, t_0=0., rootfinder_options=None):
        """Find a equilibrium point for the model.
        This method solves the root finding problem:

            f(x,y,u,t_0) = 0
            g(x,y,u,t_0) = 0
            additional_eqs (x,y,u,t_0) = 0

        Use additional_eqs to specify the additional conditions remembering that dim(additional_eqs) = n_u,
        so the system can be well defined.
        If no initial guess is provided ("guess" parameter) a guess of ones will be used (not zero to avoid problems
        with singularities.

        Returns x_0, y_0, u_0

        :param dict rootfinder_options: options to be passed to rootfinder
        :param additional_eqs: SX
        :param guess: DM
        :param t_0: float
        :return: (DM, DM, DM)
        """
        if rootfinder_options is None:
            rootfinder_options = dict(nlpsol='ipopt', nlpsol_options=config.SOLVER_OPTIONS['nlpsol_options'])
        if guess is None:
            guess = [1] * (self.n_x + self.n_y + self.n_u)
        if isinstance(additional_eqs, list):
            additional_eqs = vertcat(*additional_eqs)

        eqs = vertcat(self.ode, self.alg, additional_eqs)
        eqs = substitute(eqs, self.t_sym, t_0)
        eqs = substitute(eqs, self.tau_sym, 0)
        f_eqs = Function('f_equilibrium', [vertcat(*self.all_sym[1:-1])], [eqs])

        rf = rootfinder('rf_equilibrium', 'nlpsol', f_eqs, rootfinder_options)
        res = rf(guess)
        return res[:self.n_x], res[self.n_x:self.n_x + self.n_y], res[self.n_x + self.n_y:]


#######################################################################

class SuperModel(SystemModel):
    def __init__(self, models=None, connections=None, **kwargs):
        if models is None:
            models = []
        if connections is None:
            connections = []
        self.models = models
        SystemModel.__init__(self, **kwargs)

        connecting_equations, free_zs = zip(*connections)
        self.merge(self.models, connecting_equations=connecting_equations)
