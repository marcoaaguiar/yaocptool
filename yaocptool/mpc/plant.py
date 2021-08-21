from typing import List, Optional, Protocol, Tuple, Union

import numpy
from casadi import DM, Function, mtimes, vertcat

from yaocptool.modelling import DataSet, SystemModel


class PlantInterface(Protocol):
    name: str
    t_0: float
    t_s: float

    def get_measurement(self) -> Tuple[float, DM, DM]:
        """
        :returns: (timestamp, measuremnt, control)
        """
        ...

    def set_control(self, u):
        pass


class PlantSimulation(PlantInterface):
    """
    Simulates a plant using a model.

    """

    def __init__(
        self,
        model: SystemModel,
        x_0: DM,
        u_0: DM,
        y_guess: float,
        t_0: float = 0.0,
        t_s: float = 0.0,
        c_matrix: Optional[DM] = None,
        d_matrix: Optional[DM] = None,
        verbosity: int = 1,
        super_samling: Union[int, List[float]] = 1,
        u_function: Optional[Function] = None,
    ):
        """
            Plant which uses a SystemModel.simulate to obtain the measurements.

        :param SystemModel model: simulation model
        :param DM x_0: initial condition
        :param DM t_s: (default: 1) sampling time
        :param DM u: (default: 0) initial control
        :param DM y_guess: initial guess for algebraic variables for simulation
        :param DM t_0: (default: 0) initial time
        :param dict integrator_options: integrator options
        :param bool has_noise: Turn on/off the process/measurement noise
        :param DM r_n: Measurement noise covariance matrix
        :param DM r_v: Process noise covariance matrix
        :param noise_seed: Seed for the random number generator used to create noise. Use the same seed for the
            repeatability in the experiments.
        """
        self.model = model
        self.name = self.model.name
        self.t_s = float(t_s)

        self.x = x_0
        self.measurement = x_0
        self.u = u_0 if u_0 is not None else DM.zeros(self.model.n_u)
        self.t = self.t_0 = float(t_0)

        self.y_guess = y_guess
        self.u_function = u_function

        self.c_matrix = (
            c_matrix
            if c_matrix is not None
            else DM.eye(self.model.n_x + self.model.n_y)
        )
        self.d_matrix = d_matrix if d_matrix is not None else DM(0.0)
        self.d_matrix = None

        self.p = None
        self.theta = None

        # Noise
        self.has_noise = False
        self.r_n = DM(0.0)
        self.r_v = DM(0.0)
        self.noise_seed = None

        # Options
        self.verbosity = verbosity
        self.sampling_points = (
            [1.0 / super_samling * (point + 1) for point in range(super_samling)]
            if isinstance(super_samling, int)
            else super_samling
        )
        self.integrator_options = None
        self.dataset = DataSet(name="Plant")

        self._iterations = 0

        self._initialize_dataset()

        meas_data = self._measure(self.x, self.y_guess)

        self._save_data(self.x, self.y_guess, self.u, **meas_data)
        if self.has_noise and self.noise_seed is not None:
            numpy.random.seed(self.noise_seed)

    def get_measurement(self) -> Tuple[float, DM, DM]:
        """Return the plant measurement of a simulated model and advance time by 't_s'.
        Return the measurement time, the measurement [x; y], and the controls.

        :rtype: tuple
        :return: (timestamp, measuremnt, control)
        """
        return self.t, self.measurement, self.u

    def set_control_function(self, u_func):
        """
        Set a new control for the plant and simulate

        :param DM u: new control vector
        """
        self.u_function = u_func

        # go to next time
        self._advance()

    def set_control(self, u: Union[DM, float, List[float]]):
        """
        Set a new control for the plant and simulate

        :param DM u: new control vector
        """
        if isinstance(u, (list, int, float)):
            u = vertcat(u)

        if self.verbosity >= 1:
            print("Set control: {}".format(u))

        if u.shape[0] != self.model.n_u_par:
            raise ValueError(
                "Given control does not have the same size of the plant."
                "Plant control size: {}, given control size: {}".format(
                    self.model.n_u_par, u.shape[0]
                )
            )
        self.u = u

        # go to next time
        self._advance()

    def _advance(self):
        t_start = self.t
        for delta_t in self.sampling_points:
            t_f = t_start + delta_t * self.t_s

            if self.u_function is not None:
                self.u = self.u_function(self.t)  # type: ignore

            sim_result = self.model.simulate(
                x_0=self.x,
                t_0=self.t,
                t_f=t_f,
                y_0=self.y_guess,
                u=self.u,
                p=self.p,
                theta=self.theta,
                integrator_options=self.integrator_options,
            )

            x_sim, y, u = sim_result.final_condition().values()

            # Process noise
            x = x_sim + self._process_noise() if self.has_noise else x_sim

            self.t = t_f
            self.x = x
            self._save_data(x, y, u)

        # Measure
        meas_data = self._measure(x, y)
        self.dataset.insert_data("meas", self.t, meas_data["measurement"])
        self.dataset.insert_data(
            "meas_wo_noise", self.t, meas_data["measurement_wo_noise"]
        )
        self.measurement = meas_data["measurement"]

    def _process_noise(self):
        return DM(numpy.random.multivariate_normal([0] * self.r_v.shape[0], self.r_v))

    def _measure(self, x, y):
        measurement_wo_noise = mtimes(self.c_matrix, vertcat(x, y))
        if self.has_noise:
            n_rand = DM(
                numpy.random.multivariate_normal([0] * self.r_n.shape[0], self.r_n)
            )
            measurement = measurement_wo_noise + n_rand
        else:
            measurement = measurement_wo_noise

        return {
            "measurement": measurement,
            "measurement_wo_noise": measurement_wo_noise,
        }

    def _save_data(self, x, y, u, measurement=None, measurement_wo_noise=None):
        self.dataset.insert_data("x", self.t, x)
        self.dataset.insert_data("y", self.t, y)
        self.dataset.insert_data("u", self.t, u)

        if measurement is not None:
            self.dataset.insert_data("meas", self.t, measurement)

        if measurement is not None:
            self.dataset.insert_data("meas_wo_noise", self.t, measurement_wo_noise)

    def _initialize_dataset(self):
        self.dataset.create_entry(
            "x",
            self.model.n_x,
            [self.model.x[i].name() for i in range(self.model.n_x)],
        )

        self.dataset.create_entry(
            "y",
            self.model.n_y,
            [self.model.y[i].name() for i in range(self.model.n_y)],
        )

        self.dataset.create_entry(
            "u",
            self.model.n_u,
            [self.model.u[i].name() for i in range(self.model.n_u)],
            plot_style="step",
        )

        self.dataset.create_entry(
            "meas_wo_noise",
            self.model.n_x,
            ["meas_wo_noise_" + str(i) for i in range(self.model.n_x)],
        )
        self.dataset.create_entry(
            "meas",
            self.model.n_x,
            ["meas_" + str(i) for i in range(self.model.n_x)],
        )
