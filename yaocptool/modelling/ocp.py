# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 11:15:03 2017

@author: marco
"""

from casadi import DM, repmat, vertcat, substitute, mtimes, is_equal, SX, inf, gradient, dot, jacobian

from yaocptool import find_variables_indices_in_vector, remove_variables_from_vector_by_indices
from yaocptool.modelling import SystemModel


class OptimalControlProblem(object):
    r""" Optimal Control Problem class, used to define a optimal control problem based on a model (SystemModel)
    It has the following form:

    .. math::
       \\min J &= V(x(t_f), p) + \int_{t_0} ^{t_f} L(x,y,u,t,p,\\theta) \, dt
        + \sum_{k} S(x(t_k), y(t_k), u(t_k), t_k, p, \\theta_k)

       \\textrm{s.t.:}\,& \dot{x} = f(x,y,u,t,p,\\theta)

       & g(x,y,u,t,p,\\theta) = 0

       & g_{eq} (x,y,u,t,p,\\theta) = 0

       & g_{ineq}(x,y,u,t,p,\\theta) \leq 0

       & x_{min} \leq x \leq x_{max}

       & y_{min} \leq y \leq y_{max}

       & u_{min} \leq u \leq u_{max}

       & \Delta u_{min} \leq \Delta u \leq \Delta u_{max}

       & h_{initial} (x (t_0), t_0, p, \\theta) = 0

       & h_{final} (x (t_f), t_f, p, \\theta) = 0

       & h (p) = 0

    where  :math:`t_k` is final time in each finite element.

    """

    def __init__(self, model=None, name="OCP", x_0=None, t_0=0.0, t_f=1.0, **kwargs):
        """

        :param yaocptool.modelling.SystemModel model:
        :param str name:
        :param casadi.DM|list x_0: initial conditions
        :param float t_0:
        :param float t_f:
        :param kwargs:
        """
        if x_0 is None:
            x_0 = DM([])

        self.t_0 = t_0
        self.t_f = t_f
        self.x_0 = x_0

        self.name = name
        self._model = model  # type: SystemModel
        self.model = model.get_copy()  # type: SystemModel

        self.x_cost = None
        self.eta = SX()
        self.p_opt = vertcat([])
        self.theta_opt = vertcat([])

        self.x_max = repmat(inf, self.model.n_x)
        self.y_max = repmat(inf, self.model.n_y)
        self.u_max = repmat(inf, self.model.n_u)
        self.delta_u_max = repmat(inf, self.model.n_u)
        self.p_opt_max = repmat(inf, self.n_p_opt)
        self.theta_opt_max = repmat(inf, self.n_theta_opt)

        self.x_min = repmat(-inf, self.model.n_x)
        self.y_min = repmat(-inf, self.model.n_y)
        self.u_min = repmat(-inf, self.model.n_u)
        self.delta_u_min = repmat(-inf, self.model.n_u)
        self.p_opt_min = repmat(-inf, self.n_p_opt)
        self.theta_opt_min = repmat(-inf, self.n_theta_opt)

        self.h = vertcat([])
        self.h_initial = self.model.x_sym - self.model.x_0_sym
        self.h_final = vertcat([])

        self.g_eq = vertcat([])
        self.g_ineq = vertcat([])
        self.time_g_eq = []
        self.time_g_ineq = []

        self.L = DM(0.)  # Integral cost
        self.V = DM(0.)  # Final cost
        self.S = DM(0.)  # Finite element final cost
        self.H = DM(0.)

        self.last_u = None

        self.y_guess = None
        self.u_guess = None

        self.parametrized_control = False
        self.positive_objective = False
        self.NULL_OBJ = False

        if 'obj' in kwargs:
            obj_value = kwargs.pop('obj')
            if type(obj_value) == dict:
                for (k, v) in obj_value.items():
                    setattr(self, k, v)
                self.create_quadratic_cost(obj_value)

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        self.x_0 = DM(self.x_0)

    @property
    def n_h_final(self):
        return self.h_final.shape[0]

    @property
    def n_h_initial(self):
        return self.h_initial.shape[0]

    @property
    def n_g_ineq(self):
        return self.g_ineq.shape[0]

    @property
    def n_g_eq(self):
        return self.g_eq.shape[0]

    @property
    def n_eta(self):
        return self.eta.shape[0]

    @property
    def n_p_opt(self):
        if isinstance(self.p_opt, list):
            self.p_opt = vertcat(*self.p_opt)
        return self.p_opt.shape[0]

    @property
    def n_theta_opt(self):
        if isinstance(self.theta_opt, list):
            self.theta_opt = vertcat(*self.theta_opt)
        return self.theta_opt.shape[0]

    @property
    def has_delta_u(self):
        has_element_diff_from_inf = False
        for i in range(self.model.n_u):
            has_element_diff_from_inf = (not is_equal(self.delta_u_max[i], inf)) or has_element_diff_from_inf
            has_element_diff_from_inf = (not is_equal(self.delta_u_min[i], -inf)) or has_element_diff_from_inf
        return has_element_diff_from_inf

    def create_state(self, name, size=1, ode=None, x_0=None, x_max=None, x_min=None, h_initial=None):
        var = SX.sym(name, size)
        self.include_state(var, ode=ode, x_0=x_0, x_min=x_min, x_max=x_max, h_initial=h_initial)
        return var

    def create_algebraic(self, name, size=1, alg=None, y_max=None, y_min=None, y_guess=None):
        var = SX.sym(name, size)
        self.include_algebraic(var, alg=alg, y_min=y_min, y_max=y_max, y_guess=y_guess)
        return var

    def create_control(self, name, size=1, u_min=None, u_max=None, delta_u_min=None, delta_u_max=None, u_guess=None):
        var = SX.sym(name, size)
        self.include_control(var, u_min=u_min, u_max=u_max,
                             delta_u_min=delta_u_min, delta_u_max=delta_u_max, u_guess=u_guess)
        return var

    @property
    def create_input(self):
        return self.create_control

    def create_parameter(self, name, size=1):
        new_p = self.model.create_parameter(name=name, size=size)
        return new_p

    def create_theta(self, name, size=1):
        new_theta = self.model.create_theta(name=name, size=size)
        return new_theta

    def create_optimization_parameter(self, name, size=1, p_opt_min=None, p_opt_max=None):
        new_p_opt = self.model.create_parameter(name=name, size=size)

        self.set_parameter_as_optimization_parameter(new_p_opt, new_p_opt_min=p_opt_min, new_p_opt_max=p_opt_max)
        return new_p_opt

    def create_optimization_theta(self, name, size=1, new_theta_opt_min=None, new_theta_opt_max=None):
        new_theta_opt = self.model.create_theta(name=name, size=size)

        self.set_theta_as_optimization_theta(new_theta_opt, new_theta_opt_min=new_theta_opt_min,
                                             new_theta_opt_max=new_theta_opt_max)
        return new_theta_opt

    def create_adjoint_states(self):
        lamb = SX.sym(self.name + '_lamb', self.model.n_x)
        nu = SX.sym(self.name + '_nu', self.model.n_y)

        self.eta = SX.sym(self.name + '_eta', self.n_h_final)

        self.H = self.L + dot(lamb, self.model.ode) + dot(nu, self.model.alg)

        l_dot = -gradient(self.H, self.model.x_sym)
        alg_eq = gradient(self.H, self.model.y_sym)

        self.include_state(lamb, l_dot, suppress=True)
        self.model.has_adjoint_variables = True

        self.include_algebraic(nu, alg_eq)

        self.h_final = vertcat(self.h_final,
                               self.model.lamb_sym - gradient(self.V, self.model.x_sys_sym)
                               - mtimes(jacobian(self.h_final, self.model.x_sys_sym).T,
                                        self.eta))

    def include_system_equations(self, ode=None, alg=None):
        self.model.include_system_equations(ode=ode, alg=alg)

    def include_state(self, var, ode=None, x_0=None, x_min=None, x_max=None, h_initial=None, x_0_sym=None,
                      suppress=False):
        if x_0 is not None:
            x_0 = vertcat(x_0)
        if x_min is None:
            x_min = -DM.inf(var.numel())
        if x_max is None:
            x_max = DM.inf(var.numel())
        if x_0 is None and h_initial is None and not suppress:
            raise Exception('No initial condition given')

        var = vertcat(var)
        x_min = vertcat(x_min)
        x_max = vertcat(x_max)

        if not var.numel() == x_max.numel():
            raise ValueError('Size of "x" and "x_max" differ. x.numel()={} '
                             'and x_max.numel()={}'.format(var.numel(), x_max.numel()))
        if not var.numel() == x_min.numel():
            raise ValueError('Size of "x" and "x_min" differ. x.numel()={} '
                             'and x_min.numel()={}'.format(var.numel(), x_min.numel()))
        if not var.numel() == x_min.numel():
            raise ValueError('Size of "x" and "x_0" differ. x.numel()={} '
                             'and x_0.numel()={}'.format(var.numel(), x_0.numel()))

        x_0_sym = self.model.include_state(var, ode, x_0_sym)

        if x_0 is not None:
            self.x_0 = vertcat(self.x_0, x_0)
            h_initial = x_0_sym - var
        else:
            x_0 = DM.zeros(var.shape)
            self.x_0 = vertcat(self.x_0, x_0)

        if h_initial is not None:
            self.h_initial = vertcat(self.h_initial, h_initial)

        self.x_min = vertcat(self.x_min, x_min)
        self.x_max = vertcat(self.x_max, x_max)

    def include_algebraic(self, var, alg=None, y_min=None, y_max=None, y_guess=None):
        if y_min is None:
            y_min = -DM.inf(var.numel())
        if y_max is None:
            y_max = DM.inf(var.numel())

        if isinstance(y_min, list):
            y_min = vertcat(*y_min)
        if isinstance(y_max, list):
            y_max = vertcat(*y_max)

        if not var.numel() == y_min.numel():
            raise ValueError(
                "Given 'var' and 'y_min' does not have the same size, {}!={}".format(var.numel(), y_min.numel()))
        if not var.numel() == y_max.numel():
            raise ValueError(
                "Given 'var' and 'y_max' does not have the same size, {}!={}".format(var.numel(), y_max.numel()))

        if self.y_guess is not None:
            if y_guess is None:
                self.y_guess = vertcat(self.y_guess, DM.zeros(var.numel()))
            else:
                self.y_guess = vertcat(self.y_guess, y_guess)

        self.model.include_algebraic(var, alg)
        self.y_min = vertcat(self.y_min, y_min)
        self.y_max = vertcat(self.y_max, y_max)

    def include_control(self, var, u_min=None, u_max=None, delta_u_min=None, delta_u_max=None, u_guess=None):
        if isinstance(var, list):
            var = vertcat(*var)

        # if not given
        if u_min is None:
            u_min = -DM.inf(var.numel())
        if u_max is None:
            u_max = DM.inf(var.numel())
        if delta_u_min is None:
            delta_u_min = -DM.inf(var.numel())
        if delta_u_max is None:
            delta_u_max = DM.inf(var.numel())
        if u_guess is None and self.u_guess is not None:
            raise ValueError('The OptimalControlProblem already has a control guess ("ocp.u_guess"), but no guess was '
                             'passed for the new variable (parameter "u_guess" is None). Either remove all guesses '
                             '("ocp.u_guess = None") before including the new variable, or pass a guess for the control'
                             ' using the parameter "u_guess"')

        # if given as a list
        if isinstance(u_min, list):
            u_min = vertcat(*u_min)
        if isinstance(u_max, list):
            u_max = vertcat(*u_max)
        if isinstance(delta_u_min, list):
            delta_u_min = vertcat(*delta_u_min)
        if isinstance(delta_u_max, list):
            delta_u_max = vertcat(*delta_u_max)
        if u_guess is not None and isinstance(u_guess, list):
            u_guess = vertcat(*u_guess)

        # if not a casadi type
        u_min = vertcat(u_min)
        u_max = vertcat(u_max)
        delta_u_min = vertcat(delta_u_min)
        delta_u_max = vertcat(delta_u_max)
        if u_guess is not None:
            u_guess = vertcat(u_guess)

        # if passed a scalar but meant a vector of that scalar
        if u_min.numel() == 1 and var.numel() > 1:
            u_min = repmat(u_min, var.numel())
        if u_max.numel() == 1 and var.numel() > 1:
            u_max = repmat(u_max, var.numel())
        if delta_u_min.numel() == 1 and var.numel() > 1:
            delta_u_min = repmat(delta_u_min, var.numel())
        if delta_u_max.numel() == 1 and var.numel() > 1:
            delta_u_max = repmat(delta_u_max, var.numel())

        # if passed but has a wrong size
        if not var.numel() == u_min.numel():
            raise ValueError(
                "Given 'var' and 'u_min' does not have the same size, {}!={}".format(var.numel(), u_min.numel()))
        if not var.numel() == u_max.numel():
            raise ValueError(
                "Given 'var' and 'u_max' does not have the same size, {}!={}".format(var.numel(), u_max.numel()))
        if not var.numel() == delta_u_min.numel():
            raise ValueError(
                "Given 'var' and 'delta_u_min' does not have the same size, {}!={}".format(var.numel(),
                                                                                           delta_u_min.numel()))
        if not var.numel() == delta_u_max.numel():
            raise ValueError(
                "Given 'var' and 'delta_u_max' does not have the same size, {}!={}".format(var.numel(),
                                                                                           delta_u_max.numel()))
        if u_guess is not None and var.numel() != u_guess.numel():
            raise ValueError(
                "Given 'var' and 'u_guess' does not have the same size, {}!={}".format(var.numel(),
                                                                                       u_guess.numel()))

        self.u_min = vertcat(self.u_min, u_min)
        self.u_max = vertcat(self.u_max, u_max)
        self.delta_u_min = vertcat(self.delta_u_min, delta_u_min)
        self.delta_u_max = vertcat(self.delta_u_max, delta_u_max)

        if u_guess is not None:
            if self.model.n_u == 0:
                self.u_guess = u_guess
            else:
                self.u_guess = vertcat(self.u_guess, u_guess)

        self.model.include_control(var)

    def include_parameter(self, p):
        self.model.include_parameter(p)

    def include_theta(self, theta):
        self.model.include_theta(theta)

    def include_optimization_parameter(self, var, p_opt_min=None, p_opt_max=None):
        if p_opt_min is None:
            p_opt_min = -DM.inf(var.numel())
        if p_opt_max is None:
            p_opt_max = DM.inf(var.numel())

        self.model.include_parameter(var)
        self.set_parameter_as_optimization_parameter(var, p_opt_min, p_opt_max)

    def include_optimization_theta(self, var, theta_opt_min=None, theta_opt_max=None):
        if theta_opt_min is None:
            theta_opt_min = -DM.inf(var.numel())
        if theta_opt_max is None:
            theta_opt_max = DM.inf(var.numel())

        self.model.include_theta(var)
        self.set_theta_as_optimization_theta(var, new_theta_opt_min=theta_opt_min, new_theta_opt_max=theta_opt_max)

    def include_initial_time_equality(self, eq):
        """Include initial time equality. Equality that is evaluated at t=t_0.
        The equality is concatenated to "h_initial"

        :param eq: initial equality constraint
        """
        if isinstance(eq, list):
            eq = vertcat(*eq)
        self.h_initial = vertcat(self.h_initial, eq)

    def include_final_time_equality(self, eq):
        """Include final time equality. Equality that is evaluated at t=t_f.
        The equality is concatenated to "h_final"

        :param eq: final equality constraint
        """
        if isinstance(eq, list):
            eq = vertcat(*eq)
        self.h_final = vertcat(self.h_final, eq)

    def include_time_inequality(self, eq, when='default'):
        r"""Include time dependent inequality.
        g_ineq(..., t) <= 0, for t \in [t_0, t_f]

        The inequality is concatenated to "g_ineq"

        :param str when: Can be 'default', 'end', 'start'.
            'start' - the constraint will be evaluated at the start of every finite element
            'end' - the constraint will be evaluated at the end of every finite element
            'default' - will be evaluated at each collocation point of every finite element.
            For the multiple shooting, the constraint will be evaluated at the end of each
            finite element
        :param eq: inequality
        """
        if isinstance(eq, list):
            eq = vertcat(*eq)
        self.g_ineq = vertcat(self.g_ineq, eq)
        self.time_g_ineq.extend([when] * eq.numel())

    def include_time_equality(self, eq, when='default'):
        r"""Include time dependent equality.
        g_eq(..., t) = 0, for t \in [t_0, t_f]

        The inequality is concatenated to "g_ineq"

        :param eq: equality
        :param str when: Can be 'default', 'end', 'start'.
            'start' - the constraint will be evaluated at the start of every finite element
            'end' - the constraint will be evaluated at the end of every finite element
            'default' - will be evaluated at each collocation point of every finite element.
            For the multiple shooting, the constraint will be evaluated at the end of each
            finite element
        """
        if isinstance(eq, list):
            eq = vertcat(*eq)
        self.g_eq = vertcat(self.g_eq, eq)
        self.time_g_eq.extend([when] * eq.numel())

    def include_equality(self, eq):
        """Include time independent equality.
        Equality is concatenated "h".

        :param eq: time independent equality
        """
        if isinstance(eq, list):
            eq = vertcat(*eq)
        self.h = vertcat(self.h, eq)

    def remove_algebraic(self, var, eq=None):
        to_remove = find_variables_indices_in_vector(var, self.model.y_sym)
        to_remove.reverse()

        self.y_max = remove_variables_from_vector_by_indices(to_remove, self.y_max)
        self.y_min = remove_variables_from_vector_by_indices(to_remove, self.y_min)
        if self.y_guess is not None:
            self.y_guess = remove_variables_from_vector_by_indices(to_remove, self.y_guess)

        self.model.remove_algebraic(var, eq)

    def remove_control(self, var):
        to_remove = find_variables_indices_in_vector(var, self.model.u_sym)

        self.u_max = remove_variables_from_vector_by_indices(to_remove, self.u_max)
        self.u_min = remove_variables_from_vector_by_indices(to_remove, self.u_min)
        self.delta_u_max = remove_variables_from_vector_by_indices(to_remove, self.delta_u_max)
        self.delta_u_min = remove_variables_from_vector_by_indices(to_remove, self.delta_u_min)

        if self.u_guess is not None:
            self.u_guess = remove_variables_from_vector_by_indices(to_remove, self.u_guess)

        self.model.remove_control(var)

    def replace_variable(self, original, replacement):
        """
            Replace 'original' by 'replacement' in the problem and model equations

        :param original:
        :param replacement:
        """
        if isinstance(original, list):
            original = vertcat(*original)
        if isinstance(replacement, list):
            replacement = vertcat(*replacement)

        if not original.numel() == replacement.numel():
            raise ValueError('Size of "original" and "replacement" are not equal, {}!={}'.format(original.numel(),
                                                                                                 replacement.numel()))
        # if original.numel():
        self.L = substitute(self.L, original, replacement)
        self.V = substitute(self.V, original, replacement)
        self.S = substitute(self.S, original, replacement)

        # change its own variables
        self.h_initial = substitute(self.h_initial, original, replacement)
        self.h_final = substitute(self.h_final, original, replacement)
        self.g_ineq = substitute(self.g_ineq, original, replacement)
        self.g_eq = substitute(self.g_eq, original, replacement)
        self.h = substitute(self.h, original, replacement)

        # apply to the model
        self.model.replace_variable(original, replacement)

    def parametrize_control(self, u_sym, expr, u_par=None):
        """
            Parametrize the control variable

        :param u_sym:
        :param expr:
        :param u_par:
        """
        # parametrize on the model
        self.model.parametrize_control(u_sym=u_sym, expr=expr, u_par=u_par)

        # replace the OCP equations
        self.replace_variable(u_sym, expr)

    def _fix_types(self):
        """
            Transform attributes in casadi types.
        """
        self.x_max = vertcat(self.x_max)
        self.y_max = vertcat(self.y_max)
        self.u_max = vertcat(self.u_max)
        self.delta_u_max = vertcat(self.delta_u_max)

        self.x_min = vertcat(self.x_min)
        self.y_min = vertcat(self.y_min)
        self.u_min = vertcat(self.u_min)
        self.delta_u_min = vertcat(self.delta_u_min)

        self.h_final = vertcat(self.h_final)
        self.h_initial = vertcat(self.h_initial)
        self.h = vertcat(self.h)

        self.g_eq = vertcat(self.g_eq)
        self.g_ineq = vertcat(self.g_ineq)

        self.x_0 = vertcat(self.x_0)
        if self.y_guess is not None:
            self.y_guess = vertcat(self.y_guess)
        if self.u_guess is not None:
            self.u_guess = vertcat(self.u_guess)

        if isinstance(self.L, (int, float)):
            self.L = DM(self.L)
        if isinstance(self.V, (int, float)):
            self.V = DM(self.V)
        if isinstance(self.S, (int, float)):
            self.S = DM(self.S)

    def pre_solve_check(self):
        self._fix_types()

        # Check if Objective Function was provided
        # if self.L.is_zero() and self.V.is_zero() and self.S.is_zero() and not self.NULL_OBJ:
        #     raise Exception('No objective, if this is intentional use problem.NULL_OBJ = True')

        # Check if the objective function has the proper size
        if not self.L.numel() == 1:
            raise Exception(
                'Size of dynamic cost (ocp.L) is different from 1, provided size is: {}'.format(self.L.numel()))
        if not self.V.numel() == 1:
            raise Exception(
                'Size of final cost (ocp.V) is different from 1, provided size is: {}'.format(self.L.numel()))

        # Check if the initial condition has the same number of elements of the model
        attributes = ['x_0', 'x_max', 'y_max', 'u_max', 'x_min', 'y_min', 'u_min', 'delta_u_max',
                      'delta_u_min']
        attr_to_compare = ['n_x', 'n_x', 'n_y', 'n_u', 'n_x', 'n_y', 'n_u', 'n_u', 'n_u']
        for i, attr in enumerate(attributes):
            if not getattr(self, attr).numel() == getattr(self.model, attr_to_compare[i]):
                raise Exception('The size of "self.{}" is not equal to the size of "model.{}", '
                                '{} != {}'.format(attr, attr_to_compare[i], getattr(self, attr).numel(),
                                                  getattr(self.model, attr_to_compare[i])))

        # Check if the initial condition has the same number of elements of the model
        attributes_ocp = ['p_opt_max', 'p_opt_min', 'theta_opt_min', 'theta_opt_max']
        attr_to_compare_in_ocp = ['n_p_opt', 'n_p_opt', 'n_theta_opt', 'n_theta_opt']
        for i, attr in enumerate(attributes_ocp):
            if not getattr(self, attr).numel() == getattr(self, attr_to_compare_in_ocp[i]):
                raise Exception('The size of "self.{}" is not equal to the size of "model.{}", '
                                '{} != {}'.format(attr, attr_to_compare_in_ocp[i], getattr(self, attr).numel(),
                                                  getattr(self, attr_to_compare_in_ocp[i])))

        return True

    def reset_working_model(self):
        self.model = self._model.get_copy()

    def create_cost_state(self):
        r"""Create and state with the dynamics equal to L from \int_{t_0}^{t_f} L(...) dt:
        \dot{x}_c = L(...)

        :rtype: casadi.SX
        """
        if self.positive_objective:
            x_min = 0
        else:
            x_min = -inf

        x_c = self.create_state(self.name + '_x_c', size=1, ode=self.L, x_0=0, x_min=x_min)
        self.x_cost = x_c
        return x_c

    def create_quadratic_cost(self, par_dict):
        self.L = DM(0)
        self.V = DM(0)
        if 'x_ref' not in par_dict:
            par_dict['x_ref'] = DM.zeros(self.model.n_x)
        if 'u_ref' not in par_dict:
            par_dict['u_ref'] = DM.zeros(self.model.n_u)

        if 'Q' in par_dict:
            self.L += mtimes(mtimes((self.model.x_sym - par_dict['x_ref']).T, par_dict['Q']),
                             (self.model.x_sym - par_dict['x_ref']))

        if 'R' in par_dict:
            self.L += mtimes(mtimes((self.model.u_sym - par_dict['u_ref']).T, par_dict['R']),
                             (self.model.u_sym - par_dict['u_ref']))

        if 'Qv' in par_dict:
            self.V += mtimes(mtimes((self.model.x_sym - par_dict['x_ref']).T, par_dict['Qv']),
                             (self.model.x_sym - par_dict['x_ref']))

        if 'Rv' in par_dict:
            self.V += mtimes(mtimes(self.model.x_sym.T, par_dict['Rv']), self.model.x_sym)

    def merge(self, problems):
        """

        :param list of OptimalControlProblem problems:
        """
        for problem in problems:
            if not self.t_0 == problem.t_0:
                raise ValueError('Problems "{}" and "{}" have different "t_0", {}!={}'.format(self.name, problem.name,
                                                                                              self.t_0, problem.t_0))
            if not self.t_f == problem.t_f:
                raise ValueError('Problems "{}" and "{}" have different "t_f", {}!={}'.format(self.name, problem.name,
                                                                                              self.t_f, problem.t_f))

            self.x_0 = vertcat(self.x_0, problem.x_0)

            self.eta = vertcat(self.eta, problem.eta)
            self.p_opt = vertcat(self.p_opt, problem.p_opt)
            self.theta_opt = vertcat(self.theta_opt, problem.theta_opt)

            self.x_max = vertcat(self.x_max, problem.x_max)
            self.y_max = vertcat(self.y_max, problem.y_max)
            self.u_max = vertcat(self.u_max, problem.u_max)
            self.delta_u_max = vertcat(self.delta_u_max, problem.delta_u_max)
            self.p_opt_max = vertcat(self.p_opt_max, problem.p_opt_max)
            self.theta_opt_max = vertcat(self.theta_opt_max, problem.theta_opt_max)

            self.x_min = vertcat(self.x_min, problem.x_min)
            self.y_min = vertcat(self.y_min, problem.y_min)
            self.u_min = vertcat(self.u_min, problem.u_min)
            self.delta_u_min = vertcat(self.delta_u_min, problem.delta_u_min)
            self.p_opt_min = vertcat(self.p_opt_min, problem.p_opt_min)
            self.theta_opt_min = vertcat(self.theta_opt_min, problem.theta_opt_min)

            self.h = vertcat(self.h, problem.h)
            self.h_initial = vertcat(self.h_initial, problem.h_initial)
            self.h_final = vertcat(self.h_final, problem.h_final)

            self.g_eq = vertcat(self.g_eq, problem.g_eq)
            self.g_ineq = vertcat(self.g_ineq, problem.g_ineq)
            self.time_g_eq = self.time_g_eq + problem.time_g_eq
            self.time_g_ineq = self.time_g_ineq + problem.time_g_ineq

            self.L = self.L + problem.L
            self.V = self.V + problem.V
            self.S = self.S + problem.S

            if self.last_u is not None and problem.last_u is not None:
                self.last_u = vertcat(self.last_u, problem.last_u)
            else:
                self.last_u = None

            if self.y_guess is not None and problem.y_guess is not None:
                self.y_guess = vertcat(self.y_guess, problem.y_guess)
            else:
                self.y_guess = None

            if self.u_guess is not None and problem.u_guess is not None:
                self.u_guess = vertcat(self.u_guess, problem.u_guess)
            else:
                self.u_guess = None

            self.positive_objective = self.positive_objective and problem.positive_objective

            # Merge models
            self.model.merge([problem.model])

    def set_parameter_as_optimization_parameter(self, new_p_opt, new_p_opt_min=None, new_p_opt_max=None):
        if new_p_opt_min is None:
            new_p_opt_min = -DM.inf(new_p_opt.numel())
        if new_p_opt_max is None:
            new_p_opt_max = DM.inf(new_p_opt.numel())
        new_p_opt = vertcat(new_p_opt)
        new_p_opt_min = vertcat(new_p_opt_min)
        new_p_opt_max = vertcat(new_p_opt_max)
        if not new_p_opt.numel() == new_p_opt_max.numel():
            raise ValueError('Size of "new_p_opt" and "new_p_opt_max" differ. new_p_opt.numel()={} '
                             'and new_p_opt_max.numel()={}'.format(new_p_opt.numel(), new_p_opt_max.numel()))
        if not new_p_opt.numel() == new_p_opt_min.numel():
            raise ValueError('Size of "new_p_opt" and "new_p_opt_min" differ. new_p_opt.numel()={} '
                             'and new_p_opt_min.numel()={}'.format(new_p_opt.numel(), new_p_opt_min.numel()))

        self.p_opt = vertcat(self.p_opt, new_p_opt)
        self.p_opt_min = vertcat(self.p_opt_min, new_p_opt_min)
        self.p_opt_max = vertcat(self.p_opt_max, new_p_opt_max)
        return new_p_opt

    def set_theta_as_optimization_theta(self, new_theta_opt, new_theta_opt_min=None, new_theta_opt_max=None):
        if new_theta_opt_min is None:
            new_theta_opt_min = -DM.inf(new_theta_opt.numel())
        if new_theta_opt_max is None:
            new_theta_opt_max = DM.inf(new_theta_opt.numel())
        new_theta_opt = vertcat(new_theta_opt)
        new_theta_opt_min = vertcat(new_theta_opt_min)
        new_theta_opt_max = vertcat(new_theta_opt_max)
        if not new_theta_opt.numel() == new_theta_opt_max.numel():
            raise ValueError('Size of "new_theta_opt" and "new_theta_opt_max" differ. new_theta_opt.numel()={} '
                             'and new_theta_opt_max.numel()={}'.format(new_theta_opt.numel(),
                                                                       new_theta_opt_max.numel()))
        if not new_theta_opt.numel() == new_theta_opt_min.numel():
            raise ValueError('Size of "new_theta_opt" and "new_theta_opt_max" differ. new_theta_opt.numel()={} '
                             'and new_theta_opt_min.numel()={}'.format(new_theta_opt.numel(),
                                                                       new_theta_opt_min.numel()))

        self.theta_opt = vertcat(self.theta_opt, new_theta_opt)
        self.theta_opt_min = vertcat(self.theta_opt_min, new_theta_opt_min)
        self.theta_opt_max = vertcat(self.theta_opt_max, new_theta_opt_max)
        return new_theta_opt

    def get_p_opt_indices(self):
        return find_variables_indices_in_vector(self.p_opt, self.model.p_sym)

    def get_theta_opt_indices(self):
        return find_variables_indices_in_vector(self.theta_opt, self.model.theta_sym)

    def connect(self, u, y):
        """
        Connect an input 'u' to a algebraic variable 'y', u = y.
        The function will perform the following actions:
        - include an algebraic equation u - y = 0
        - remove 'u' from the input vector (since it is not a free variable anymore)
        - include 'u' into the algebraic vector, since it is an algebraic variable now.
        - move associated bounds (u_max, u_min) to (y_max, y_min), remove delta_u_max/delta_u_min

        :param u: input variable
        :param y: algebraic variable
        """
        # fix types
        if isinstance(u, list):
            u = vertcat(*u)
        if isinstance(y, list):
            y = vertcat(*y)

        # check if same size
        if not u.numel() == y.numel():
            raise ValueError('Size of "u" and "y" are not the same, u={} and y={}'.format(u.numel(), y.numel()))

        ind_u = find_variables_indices_in_vector(u, self.model.u)

        self.model.connect(u, y)

        self.y_max = vertcat(self.y_max, self.u_max[ind_u])
        self.y_min = vertcat(self.y_min, self.u_min[ind_u])

        if self.u_guess is not None and self.y_guess is not None:
            self.y_guess = vertcat(self.y_guess, self.u_guess[ind_u])

        self.u_max = remove_variables_from_vector_by_indices(ind_u, self.u_max)
        self.u_min = remove_variables_from_vector_by_indices(ind_u, self.u_min)
        if self.u_guess is not None:
            self.u_guess = remove_variables_from_vector_by_indices(ind_u, self.u_guess)

        self.delta_u_max = remove_variables_from_vector_by_indices(ind_u, self.delta_u_max)
        self.delta_u_min = remove_variables_from_vector_by_indices(ind_u, self.delta_u_min)
