from casadi import vertcat, integrator, SX, substitute, depends_on

from yaocptool import convert_expr_from_tau_to_time, config, find_variables_indices_in_vector


class DAESystem(object):
    def __init__(self, x=vertcat([]), y=vertcat([]), p=vertcat([]), ode=vertcat([]), alg=vertcat([]), t=None,
                 tau=None, **kwargs):
        """
            DAE System class used primarly for simulation by the SystemModel class

        For modelling it is recommended the use of the SystemModel class, use this class if you need more control of
        the integrator.

        :param SX ode: ODE equations
        :param SX alg: Algebraic equations
        :param SX x: State variables
        :param SX y: Algebraic variables
        :param SX p: Parameters
        :param SX t: Time variable
        :param SX tau: Tau variable
        """

        self.x = x
        self.y = y
        self.p = p
        self.ode = ode
        self.alg = alg
        self.t = t
        self.tau = tau

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

        :param casadi.SX var:
        :rtype: bool
        """
        ind = find_variables_indices_in_vector(var, vertcat(self.x, self.y, self.p, self.t, self.tau))
        return len(ind) > 0

    def depends_on(self, var):
        """
            Return True if the system of equations ('ode' and 'alg')depends on 'var' (contains 'var' in the equations).

        :param casadi.SX var:
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

        # Integrator Function name
        if 'name' in options:
            name = options.pop('name')
        else:
            name = 'integrator'

        # Load default options from the config file, if a 'options' dict was passed use that to override the default.
        for k in config.INTEGRATOR_OPTIONS:
            if k not in options:
                options[k] = config.INTEGRATOR_OPTIONS[k]

        # For backwards compatibility, previously option 'explicit' was a custom Runge-Kutta. CasADi now has it built-in
        if integrator_type == 'explicit':
            integrator_type = 'rk'

        # Return the specified integrator
        if integrator_type == 'implicit' and self.is_ode:
            return integrator(name, "cvodes", self.dae_system_dict, options)
        if integrator_type == 'implicit' and self.is_dae:
            return integrator(name, "idas", self.dae_system_dict, options)
        if integrator_type in ['rk', 'collocation', 'cvodes', 'idas']:
            return integrator(name, integrator_type, self.dae_system_dict, options)
        raise ValueError("'integrator_type'={} not available. Options available are: 'cvodes', 'idas', implicit "
                         "(default, auto-select between 'idas' and 'cvodes'), 'rk',"
                         " and 'collocation'.".format(integrator_type))
