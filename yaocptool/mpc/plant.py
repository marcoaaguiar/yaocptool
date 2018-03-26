from casadi import DM, vertcat

from yaocptool.modelling import SystemModel


class Plant:
    def __init__(self):
        self.name = 'Plant'
        pass

    def get_measurement(self):
        pass

    def set_control(self, u):
        pass


class PlantSimulation(Plant):
    """Simulates a plant using a model.

    """

    def __init__(self, model, x_0, **kwargs):
        """

        :type model: SystemModel
        :type DM x_0:
        """
        Plant.__init__(self)
        self.model = model
        self.name = self.model.name

        self.x = x_0
        self.u = DM.zeros(self.model.n_u)

        self.y_guess = None

        self.t = 0.
        self.t_s = 1.
        self.integrator_options = None

        self.simulation_results = None

        if 't_0' in kwargs:
            self.t = kwargs['t_0']

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def get_measurement(self):
        """Return the plant measurement of a simulated model and advance time by 't_s'.
        Return the measurement time, the measurement [x; y], and the controls.

        :return: tuple
        """
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
        print(u, type(u))
        if not u.size1() == self.model.n_u:
            raise ValueError("Given control does not have the same size of the plant."
                             "Plant control size: {}, given control size: {}".format(self.model.n_u, u.size1()))
        self.u = u
