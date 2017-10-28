# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 10:50:48 2016

@author: marco
"""
from warnings import warn

from casadi import SX, DM, vertcat, substitute, Function, integrator, jacobian, mtimes, rootfinder
from yaocptool import config, find_variables_indices_in_vector


# TODO: Check linearize method
# TODO: Create find_equilibrium method


class SystemModel:
    def __init__(self, n_x=0, n_y=0, n_z=0, n_u=0, n_p=0, n_theta=0, **kwargs):
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

        # Number of states
        # Number of (internal) algebraic
        # Number of external algebraic
        # Number of control
        # Number of parameters
        # Number of parameters that depend on the finite element

        self.ode = vertcat([])  # ODE
        self.alg = vertcat([])  # Algebraic equations
        self.alg_z = vertcat([])  # Algebraic equations of the z variable
        self.con = vertcat([])  # Connecting algebraic equations
        self.relaxed_alg = vertcat([])

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if not hasattr(self, 'name'):
            self.name = 'model'
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
        self.z_sym = SX.sym(self.name + '_z', n_z)
        self.u_sym = SX.sym(self.name + '_u', n_u)
        self.p_sym = SX.sym(self.name + '_p', n_p)
        self.theta_sym = SX.sym(self.name + '_theta', n_theta)

        self.t_sym = SX.sym('t')
        self.tau_sym = SX.sym('tau')

        self.u_par = vertcat(self.u_sym)
        self.u_func = vertcat(self.u_sym)

        self.hasAdjointVariables = False
        self.con_z = []

    @property
    def system_type(self):
        if self.n_y + self.n_z > 0:
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
    def n_z(self):
        return self.z_sym.numel()

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
    def n_yz(self):
        return self.n_y + self.n_z

    @property
    def yz_sym(self):
        return vertcat(self.y_sym, self.z_sym)

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
        return self.t_sym, self.x_sym, self.y_sym, self.z_sym, self.u_sym, self.p_sym, self.theta_sym, self.u_par

    @property
    def all_alg(self):
        return vertcat(self.alg, self.alg_z, self.con)

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
        s += 'Number of ext. algebraic (z): {:4} | Number of controls (u):                {:4}'.format(self.n_z,
                                                                                                       self.n_u)
        s += '\n'
        s += 'Number of parameters (p):     {:4} | Number of finite elem. param. (theta): {:4}'.format(self.n_p,
                                                                                                       self.n_theta)
        s += '\n'
        s += '-' * 20 + '\n'
        s += 'Number of ODE:                {:4} | Number of algebraic eq.:               {:4}'.format(
            self.ode.numel(),
            self.alg.numel())
        s += '\n'
        s += 'Number of external alg. eq.:  {:4} | Number of connecting eq.:              {:4}'.format(
            self.alg_z.numel(), self.con.numel())
        s += '\n'
        s += '=' * 20 + '\n'
        return s

    def replace_variable(self, original, replacement, variable_type='other'):
        """
            Replace a variable or parameter by an variable or expression.
            :param replacement:
            :param variable_type:
            :param original: SX: and replacement, and also variable type which
            describes which type of variable is being remove to it from the
            counters. Types: 'x', 'y', 'u', 'p', 'ignore'
        """

        self.ode = substitute(self.ode, original, replacement)
        self.alg = substitute(self.alg, original, replacement)
        self.alg_z = substitute(self.alg_z, original, replacement)
        self.con = substitute(self.con, original, replacement)
        self.relaxed_alg = substitute(self.relaxed_alg, original, replacement)

        if variable_type == 'u':
            pass
        elif variable_type == 'other':
            pass
        else:
            raise Exception('Not implemented')

    # region INCLUDES

    def include_system_equations(self, ode=None, alg=None, alg_z=None, con=None):
        if ode is None:
            ode = []
        elif isinstance(ode, list):  # if ode is list or tuple
            ode = vertcat(*ode)

        if alg is None:
            alg = []
        elif isinstance(alg, list):  # if alg is list or tuple
            alg = vertcat(*alg)

        if alg_z is None:
            alg_z = []
        elif isinstance(alg_z, list):  # if alg_z is list or tuple
            alg_z = vertcat(*alg_z)

        if con is None:
            con = []
        elif isinstance(con, list):  # if con is list or tuple
            con = vertcat(*con)

        self.ode = vertcat(self.ode, ode)
        self.alg = vertcat(self.alg, alg)
        self.alg_z = vertcat(self.alg_z, alg_z)
        self.con = vertcat(self.con, con)

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

    def include_algebraic(self, var, alg):
        self.y_sym = vertcat(self.y_sym, var)
        self.alg = vertcat(self.alg, alg)

    def include_external_algebraic(self, var, alg_z=None):
        if alg_z is None:
            alg_z = []
        self.z_sym = vertcat(self.z_sym, var)
        self.alg_z = vertcat(self.alg_z, alg_z)

    def include_connecting_equations(self, con, con_z=None):
        if con_z is None:
            con_z = []
        self.con = vertcat(self.con, con)
        self.con_z = vertcat(self.con_z, con_z)

    def include_control(self, var):
        self.u_sym = vertcat(self.u_sym, var)
        self.u_par = vertcat(self.u_par, var)

    def include_parameter(self, p):
        self.p_sym = vertcat(self.p_sym, p)

    def include_theta(self, theta):
        self.theta_sym = vertcat(self.theta_sym, theta)

    # endregion

    # region REMOVE

    def remove_variables_from_vector(self, var, vector):
        to_remove = self.find_variables_indices_in_vector(var, vector)
        to_remove.sort(reverse=True)
        for it in to_remove:
            vector.remove([it], [])
        return vector

    def remove_algebraic(self, var, eq=None):
        self.remove_variables_from_vector(var, self.y_sym)
        if eq is not None:
            self.remove_variables_from_vector(eq, self.alg)

    def remove_external_algebraic(self, var, eq=None):
        self.remove_variables_from_vector(var, self.z_sym)
        if eq is not None:
            self.remove_variables_from_vector(eq, self.alg_z)

    def remove_connecting_equations(self, var, eq):
        self.remove_variables_from_vector(var, self.z_sym)
        self.remove_variables_from_vector(eq, self.con)

    def remove_control(self, var):
        to_remove = self.find_variables_indices_in_vector(var, self.u_sym)
        to_remove.sort(reverse=True)

        for it in to_remove:
            self.u_sym.remove([it], [])
            self.u_par.remove([it], [])

    # endregion
    # ==============================================================================
    # region # Standard Function Call
    # ==============================================================================

    def slice_yz_to_y_and_z(self, yz):
        return yz[:self.n_y], yz[self.n_y:]

    @staticmethod
    def concat_y_and_z(y, z):
        return vertcat(y, z)

    @staticmethod
    def put_values_in_all_sym_format(t=None, x=None, y=None, z=None, u=None, p=None, theta=None, u_par=None):
        if t is None:
            t = []
        if x is None:
            x = []
        if y is None:
            y = []
        if z is None:
            z = []
        if u is None:
            u = []
        if p is None:
            p = []
        if theta is None:
            theta = []
        if u_par is None:
            u_par = []
        return t, x, y, z, u, p, theta, u_par

    # endregion
    # ==============================================================================

    # ==============================================================================
    # region # TIME
    # ==============================================================================

    def convert_from_time_to_tau(self, dae_sys, t_k, t_kp1):
        raise Exception('Method not implemented')

    def convert_expr_from_tau_to_time(self, expr, t_k, t_kp1):
        t = self.t_sym
        tau = self.tau_sym
        h = t_kp1 - t_k
        return substitute(expr, tau, (t - t_k) / h)

    def convert_dae_sys_from_tau_to_time(self, dae_sys, t_k, t_kp1):
        dae_sys['ode'] = self.convert_expr_from_tau_to_time(dae_sys['ode'], t_k, t_kp1)
        if 'alg' in dae_sys:
            dae_sys['alg'] = self.convert_expr_from_tau_to_time(dae_sys['alg'], t_k, t_kp1)

    # endregion
    # ==============================================================================

    # ==============================================================================
    # region MERGE
    # ==============================================================================

    def merge(self, models_list, connecting_equations=None, associated_z=None):
        if connecting_equations is None:
            connecting_equations = []
        if associated_z is None:
            associated_z = []
        if not isinstance(models_list, list):
            models_list = [models_list]

        for model in models_list:
            self.include_state(model.x_sym, model.ode, model.x_0_sym)
            self.include_algebraic(model.y_sym, model.alg)
            self.include_external_algebraic(model.z_sym, model.alg_z)
            self.include_control(model.u_sym)
            self.include_parameter(model.p_sym)
            self.include_theta(model.theta_sym)

        self.include_connecting_equations(connecting_equations, associated_z)

    # endregion
    # ==============================================================================

    # ==============================================================================
    # region SIMULATION
    # ==============================================================================

    def get_dae_system(self):
        if self.system_type == 'ode':
            system = {'x': self.x_sym, 'ode': self.ode, 't': self.t_sym}
        else:
            system = {'x': self.x_sym, 'z': vertcat(self.y_sym, self.z_sym), 'ode': self.ode,
                      'alg': vertcat(self.alg, self.alg_z, self.con), 't': self.t_sym}
        if self.n_p + self.n_theta + self.u_par.numel() > 0:
            system['p'] = vertcat(self.p_sym, self.theta_sym, self.u_par)
        return system

    def simulate(self, x_0, t_f, t_0=0, p=None, integrator_type='implicit'):
        if p is None:
            p = []
        dae = self.get_dae_system()

        opts = {'tf': t_f, 't0': t_0}  # final time
        integrator_ = self._create_integrator(dae, opts, integrator_type)
        call = {'x0': x_0, 'p': p}

        return integrator_(**call)

    def simulate_step(self, x_0, t_0, t_f, p=None, dae_sys=None, integrator_type='implicit'):
        if dae_sys is None:
            dae_sys = self.get_dae_system()

        opts = {'tf': float(t_f), 't0': float(t_0)}  # final time
        integrator_ = self._create_integrator(dae_sys, opts, integrator_type)
        args = {'x0': x_0, 'p': p}

        return integrator_(**args)

    def simulate_interval(self, x_0, t_0, t_grid, p=None, dae_sys=None, integrator_type='implicit'):
        if dae_sys is None:
            dae_sys = self.get_dae_system()
        x_vars = []
        y_vars = []
        for t in t_grid:
            opts = {'tf': float(t), 't0': float(t_0)}  # final time
            integrator_ = self._create_integrator(dae_sys, opts, integrator_type)
            call = {'x0': x_0, 'p': p}

            if p is not None:
                call['p'] = DM(p)

            res = integrator_(**call)
            x_vars.append(res['xf'])
            y_vars.append(res['zf'])
        return x_vars, y_vars

    def _create_integrator(self, dae_sys, options, integrator_type='implicit'):
        for k in config.INTEGRATOR_OPTIONS:
            options[k] = config.INTEGRATOR_OPTIONS[k]

        if integrator_type == 'implicit':
            if self.system_type == 'ode':
                integrator_ = integrator("integrator", "cvodes", dae_sys, options)
            else:
                integrator_ = integrator("integrator", "idas", dae_sys, options)
        else:
            if self.system_type == 'ode':
                integrator_ = self._create_explicit_integrator('explicitIntegrator', 'rk4', dae_sys, options)
            else:
                raise Exception('explicit integrator not implemented')
        return integrator_

    @staticmethod
    def _create_explicit_integrator(name, integrator_type, dae_sys, options=None):
        default_options = {'t0': 0, 'tf': 1, 'iterations': 4}
        if options is None:
            options = default_options
        for k in default_options:
            if k not in options:
                options[k] = default_options[k]

        if 'alg' in dae_sys:
            raise Exception('Explicit integrator not implemented for DAE systems')
        f_in = [dae_sys['t'], dae_sys['x']]
        if 'p' in dae_sys:
            f_in.append(dae_sys['p'])
        else:
            f_in.append(SX.sym('fake_p'))
        f = Function(name, f_in, [dae_sys['ode']])

        t_0 = options['t0']
        t_f = options['tf']
        iterations = options['iterations']
        n_states = dae_sys['x'].numel()
        if integrator_type == 'rk4':
            def runge_kutta_4th_order(x0=DM.zeros(n_states, 1), p=None, n_iter=iterations):
                if n_iter < 1:
                    raise Exception(
                        "The given number of Runge Kutta iterations is less than one, given {}".format(n_iter))
                if p is None:
                    p = []

                x_f = x0
                h = (t_f - t_0) / n_iter
                t = t_0
                for it in range(n_iter):
                    k1 = h * f(t, x0, p)
                    k2 = h * f(t + 0.5 * h, x0 + 0.5 * k1, p)
                    k3 = h * f(t + 0.5 * h, x0 + 0.5 * k2, p)
                    k4 = h * f(t + h, x0 + k3, p)

                    x_f = x0 + 1 / 6. * k1 + 1 / 3. * k2 + 1 / 3. * k3 + 1 / 6. * k4
                    x0 = x_f
                    t += h
                return {'xf': x_f, 'zf': []}

            return runge_kutta_4th_order

    # endregion

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

    def find_equilibrium(self, additional_eqs, guess=None, t_0=0):
        if guess is None:
            guess = [2] * (self.n_x + self.n_y + self.n_u)
        if isinstance(additional_eqs, list):
            additional_eqs = vertcat(*additional_eqs)

        eqs = vertcat(self.ode, self.alg, additional_eqs)
        eqs = substitute(eqs, self.t_sym, t_0)
        eqs = substitute(eqs, self.tau_sym, 0)
        f_eqs = Function('f_equilibrium', [vertcat(*self.all_sym[1:-1])], [eqs])
        f_jeqs = Function('f_equilibrium', [vertcat(*self.all_sym[1:-1])], [jacobian(eqs, vertcat(*self.all_sym[1:-1]))])

        rf = rootfinder('rf_equilibrium', 'newton', f_eqs)
        res = rf(guess)
        return res[:self.n_x], res[self.n_x:self.n_x + self.n_y], res[self.n_x + self.n_y:]

    @staticmethod
    def find_variables_indices_in_vector(var, vector):
        warn('Use yaocptool.find_variables_indices_in_vector')
        index = find_variables_indices_in_vector(var, vector)
        return index


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
