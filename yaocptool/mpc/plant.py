from casadi import DM, vertcat

from yaocptool.modelling import SystemModel


class Plant:
    def __init__(self):
        self.name = 'Plant'
        self._n_x = None
        pass

    @property
    def n_x(self):
        return self._n_x

    @n_x.setter
    def n_x(self, value):
        self._n_x = value

    def get_measurement(self):
        pass

    def set_control(self, u):
        pass


class PlantSimulation(Plant):
    """Simulates a plant using a model.

    """

    def __init__(self, model, x_0, **kwargs):
        """

        :param SystemModel model: simulation model
        :param DM x_0: initial condition
        :param DM t_s: (default: 0) sampling time
        :param DM u: (default: 0) initial control
        :param DM y_guess: initial guess for algebraic variables for simulation
        :param DM t_0: (default: 0) initial time
        :param dict integrator_options: integrator options
        """
        Plant.__init__(self)
        self.model = model
        self.name = self.model.name

        self.x = x_0
        self.u = DM.zeros(self.model.n_u)

        self.y_guess = None

        self.t = 0.
        self.t_s = 1.

        if self.c_matrix is None:
            self.c_matrix = DM.eye(self.model.n_x + self.model.n_y)
        if self.d_matrix is None:
            self.d_matrix = DM(0.)

        self.integrator_options = None

        self.simulation_results = None

        if 't_0' in kwargs:
            self.t = kwargs['t_0']

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    @property
    def n_x(self):
        return self.model.n_x

    def get_measurement(self):
        """Return the plant measurement of a simulated model and advance time by 't_s'.
        Return the measurement time, the measurement [x; y], and the controls.

        :return: tuple
        """
        # perform the simulation
        sim_result = self.model.simulate(x_0=self.x,
                                         t_0=self.t,
                                         t_f=self.t + self.t_s,
                                         y_0=self.y_guess,
                                         u=self.u,
                                         integrator_options=self.integrator_options)

        self.t += self.t_s
        x, y, u = sim_result.final_condition()
        self.x = x

        if self.simulation_results is None:
            self.simulation_results = sim_result
        else:
            self.simulation_results.extend(sim_result)

        return self.t, vertcat(x, y), self.u

    def set_control(self, u):
        """set a new control for the plant

        :param DM u: new control vector
        """
        if isinstance(u, list):
            u = vertcat(u)

        if not u.size1() == self.model.n_u:
            raise ValueError("Given control does not have the same size of the plant."
                             "Plant control size: {}, given control size: {}".format(self.model.n_u, u.size1()))
        self.u = u
