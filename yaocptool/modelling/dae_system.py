from casadi import vertcat, integrator, Function, SX, DM, substitute, depends_on

from yaocptool import convert_expr_from_tau_to_time, config, find_variables_indices_in_vector


class DAESystem:
    def __init__(self, **kwargs):
        """
            DAE System class used primarly for simulation by the SystemModel class

        For modelling it is recommended the use of the SystemModel class, use this class if you need more control of
        the integrator.

        :param SX ode: ODE equations
        :param alg: Algebraic equations
        :param x: State variables
        :param y: Algebraic variables
        :param p: Parameters
        :param t: Time variable
        :param tau: Tau variable
        """

        self.x = vertcat([])
        self.y = vertcat([])
        self.p = vertcat([])
        self.ode = vertcat([])
        self.alg = vertcat([])
        self.t = None
        self.tau = None
        self._integrator_is_valid = False

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    @property
    def is_dae(self):
        return self.type == 'dae'

    @property
    def is_ode(self):
        return self.type == 'ode'

    @property
    def type(self):
        if self.alg.numel() == 0:
            return 'ode'
        else:
            return 'dae'

    @property
    def has_parameters(self):
        return self.p.numel() > 0

    @property
    def dae_system_dict(self):
        if self.is_ode:
            dae_sys_dict = {'x': self.x, 'ode': self.ode, 't': self.t}
        else:
            dae_sys_dict = {'x': self.x, 'z': self.y, 'ode': self.ode, 'alg': self.alg, 't': self.t}

        if self.has_parameters:
            dae_sys_dict['p'] = self.p
        return dae_sys_dict

    def has_variable(self, var):
        """
            Return True if the var is one of the system variables (x, y, p, t, tau)

        :param SX var:
        :rtype: bool
        """
        ind = find_variables_indices_in_vector(var, vertcat(self.x, self.y, self.p, self.t, self.tau))
        return len(ind) > 0

    def depends_on(self, var):
        """
            Return True if the system of equations ('ode' and 'alg')depends on 'var' (contains 'var' in the equations).

        :param SX var:
        :rtype: bool
        """
        return depends_on(vertcat(self.ode, self.alg), var)

    def convert_from_tau_to_time(self, t_k, t_kp1):
        """
            Transform a dependence in tau into a dependence into t

        Uses the formula tau_sym = (t - t_k)/ (t_kp1 - t_k)

        :param t_k: t(k), the time at the beginning of the simulation interval
        :param t_kp1: t(k+1), the time at the end of the simulation interval
        """
        if self.t is None:
            raise AttributeError("DAESystem.t was not set: self.t = {}".format(self.t))
        if self.tau is None:
            raise AttributeError("DAESystem.t was not set: self.tau = {}".format(self.tau))

        self.alg = convert_expr_from_tau_to_time(expr=self.alg, t_sym=self.t, tau_sym=self.tau, t_k=t_k, t_kp1=t_kp1)
        self.ode = convert_expr_from_tau_to_time(expr=self.ode, t_sym=self.t, tau_sym=self.tau, t_k=t_k, t_kp1=t_kp1)

    def substitute_variable(self, old_var, new_var):
        self.ode = substitute(self.ode, old_var, new_var)
        self.alg = substitute(self.alg, old_var, new_var)
        self.x = substitute(self.x, old_var, new_var)
        self.y = substitute(self.y, old_var, new_var)
        self.p = substitute(self.p, old_var, new_var)

    def join(self, dae_sys):
        self.ode = vertcat(self.ode, dae_sys.ode)
        self.alg = vertcat(self.alg, dae_sys.alg)
        self.x = vertcat(self.x, dae_sys.x)
        self.y = vertcat(self.y, dae_sys.y)
        self.p = vertcat(self.p, dae_sys.p)

        self.substitute_variable(dae_sys.t, self.t)
        self.substitute_variable(dae_sys.tau, self.tau)

    def simulate(self, x_0, t_f, t_0=0, p=None, y_0=None, integrator_type='implicit', integrator_options=None):
        if t_f == t_0:
            raise ValueError("Initial time and final time must be different, t_0!=t_f. t_0={}, t_f={}".format(t_0, t_f))

        if self.depends_on(self.tau):
            raise AttributeError("The system of equations ('ode' and 'alg') depend on the variable 'tau'. Before being"
                                 "able to simulate it is required to transform the dependence on tau into a dependence "
                                 "on t. Use the 'convert_from_tau_to_time' for this.")

        if integrator_options is None:
            integrator_options = {}
        if p is None:
            p = []

        opts = {'tf': t_f, 't0': t_0}  # final time

        for k in integrator_options:
            opts[k] = integrator_options[k]

        call = {'x0': x_0, 'p': p}
        if self.is_dae and y_0 is not None:
            call['z0'] = y_0

        integrator_ = self._create_integrator(opts, integrator_type)

        return integrator_(**call)

    def _create_integrator(self, options=None, integrator_type='implicit'):
        if options is None:
            options = {}

        if 'name' in options:
            name = options.pop('name')
        else:
            name = 'integrator'
        for k in config.INTEGRATOR_OPTIONS:
            if k not in options:
                options[k] = config.INTEGRATOR_OPTIONS[k]

        if (integrator_type == 'implicit' and self.is_ode) or integrator_type == 'cvodes':
            return integrator(name, "cvodes", self.dae_system_dict, options)
        elif (integrator_type == 'implicit' and self.is_dae) or integrator_type == 'idas':
            return integrator(name, "idas", self.dae_system_dict, options)
        elif integrator_type == 'rk':
            return integrator(name, "rk", self.dae_system_dict, options)
        elif integrator_type == 'collocation':
            return integrator(name, "collocation", self.dae_system_dict, options)
        elif integrator_type == 'explicit':
            if self.is_ode:
                return self._create_explicit_integrator('explicit_integrator', 'rk4', self.dae_system_dict,
                                                        options)
            else:
                raise Exception('Explicit integrator not implemented for DAE systems')
        else:
            raise ValueError("'integrator_type'={} not available. Options available are: 'cvodes', 'idas', implicit "
                             "(default, auto-select between 'idas' and 'cvodes'), 'rk', 'collocation', 'explicit' "
                             "(own 4th order Runge-Kutta implementation).".format(integrator_type))

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
                return {'xf': x_f, 'zf': DM([])}

            return runge_kutta_4th_order
