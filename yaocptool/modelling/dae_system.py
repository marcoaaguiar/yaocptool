from casadi import vertcat, integrator, Function, SX, DM, substitute

from yaocptool import convert_expr_from_tau_to_time, config


class DAESystem:
    def __init__(self, **kwargs):
        self.ode = vertcat([])
        self.alg = vertcat([])
        self.x = vertcat([])
        self.y = vertcat([])
        self.p = vertcat([])
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

    def convert_from_tau_to_time(self, t_k, t_kp1):
        if self.t is None:
            raise Exception("DAESystem.t was not set: self.t = {}".format(self.t))
        if self.tau is None:
            raise Exception("DAESystem.t was not set: self.tau = {}".format(self.tau))

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

    ##############
    #  Simulate  #
    ##############

    def simulate(self, x_0, t_f, t_0=0, p=None, y_0=None, integrator_type='implicit', integrator_options=None):
        if t_f == t_0:
            raise ValueError("Initial time and final time must be different, t_0!=t_f. t_0={}, t_f={}".format(t_0, t_f))

        if integrator_options is None:
            integrator_options = {}
        if p is None:
            p = []

        opts = {'tf': t_f, 't0': t_0}  # final time
        for k in integrator_options:
            opts[k] = integrator_options[k]

        integrator_ = self._create_integrator(opts, integrator_type)
        call = {'x0': x_0, 'p': p}
        if y_0 is not None:
            call['z0'] = y_0

        return integrator_(**call)

    def _create_integrator(self, options=None, integrator_type='implicit'):
        if options is None:
            options = {}

        for k in config.INTEGRATOR_OPTIONS:
            if k not in options:
                options[k] = config.INTEGRATOR_OPTIONS[k]

        if integrator_type == 'implicit':
            if self.is_ode:
                integrator_ = integrator("integrator", "cvodes", self.dae_system_dict, options)
            else:
                integrator_ = integrator("integrator", "idas", self.dae_system_dict, options)
        else:
            if self.is_ode:
                integrator_ = self._create_explicit_integrator('explicitIntegrator', 'rk4', self.dae_system_dict,
                                                               options)
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
