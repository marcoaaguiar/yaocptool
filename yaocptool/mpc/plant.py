from casadi import DM, vertcat, mtimes
import numpy
from yaocptool.modelling import SystemModel, DataSet

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

        self.c_matrix = None
        self.d_matrix = None

        self.p = None
        self.theta = None

        # Noise
        self.has_noise = False
        self.r_n = DM(0.)
        self.r_v = DM(0.)

        self.integrator_options = None

        self.simulation_results = None
        self.dataset = DataSet(name='Plant')

        self.seed = None

        if 't_0' in kwargs:
            self.t = kwargs['t_0']

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if self.c_matrix is None:
            self.c_matrix = DM.eye(self.model.n_x + self.model.n_y)
        if self.d_matrix is None:
            self.d_matrix = DM(0.)

        self._initialize_dataset()

        if self.has_noise:
            if self.seed is not None:
                numpy.random.seed(self.seed)
            self._include_noise_in_the_model()

    def _initialize_dataset(self):
        self.dataset.data['x']['size'] = self.model.n_x
        self.dataset.data['x']['names'] = [self.model.x_sym[i].name() for i in range(self.model.n_x)]

        self.dataset.data['y']['size'] = self.model.n_y
        self.dataset.data['y']['names'] = [self.model.y_sym[i].name() for i in range(self.model.n_y)]

        self.dataset.data['u']['size'] = self.model.n_u
        self.dataset.data['u']['names'] = [self.model.u_sym[i].name() for i in range(self.model.n_u)]

        self.dataset.data['meas']['size'] = self.c_matrix.size1()
        self.dataset.data['meas']['names'] = ['meas_' + str(i) for i in range(self.model.n_x)]

    @property
    def n_x(self):
        return self.model.n_x

    def get_measurement(self):
        """Return the plant measurement of a simulated model and advance time by 't_s'.
        Return the measurement time, the measurement [x; y], and the controls.

        :return: tuple
        """
        # perform the simulation
        if self.has_noise:
            v_rand = DM(numpy.random.multivariate_normal([0] * self.r_v.size1(), self.r_v))
            if self.p is None:
                p = v_rand
            else:
                p = vertcat(self.p, v_rand)
        else:
            p = self.p

        sim_result = self.model.simulate(x_0=self.x,
                                         t_0=self.t,
                                         t_f=self.t + self.t_s,
                                         y_0=self.y_guess,
                                         u=self.u,
                                         p=p,
                                         theta=self.theta,
                                         integrator_options=self.integrator_options)

        x, y, u = sim_result.final_condition()
        self.t += self.t_s
        self.x = x
        measurement_wo_noise = mtimes(self.c_matrix, vertcat(x, y))
        if self.has_noise:
            n_rand = DM(numpy.random.multivariate_normal([0] * self.r_n.size1(), self.r_n))
            measurement = measurement_wo_noise + n_rand
        else:
            measurement = measurement_wo_noise

        self.dataset.insert_data('x', x, self.t)
        self.dataset.insert_data('y', y, self.t)
        self.dataset.insert_data('u', u, self.t)
        self.dataset.insert_data('meas', measurement, self.t)
        self.dataset.insert_data('meas_wo_noise', measurement_wo_noise, self.t)

        print('Real state: {}'.format(x))

        if self.simulation_results is None:
            self.simulation_results = sim_result
        else:
            self.simulation_results.extend(sim_result)

        return self.t, measurement, self.u

    def set_control(self, u):
        """set a new control for the plant

        :param DM u: new control vector
        """
        if isinstance(u, (list, int, float)):
            u = vertcat(u)

        if not u.shape[0] == self.model.n_u:
            raise ValueError("Given control does not have the same size of the plant."
                             "Plant control size: {}, given control size: {}".format(self.model.n_u, u.shape[0]))
        self.u = u

    def _include_noise_in_the_model(self):
        v = self.model.create_parameter('v', self.r_v.size1())
        self.model.ode[:self.r_v.size1()] = self.model.ode[:self.r_v.size1()] + v
