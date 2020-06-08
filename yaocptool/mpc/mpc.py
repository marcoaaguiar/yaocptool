import itertools
import time

from casadi import vertcat, DM

from yaocptool.estimation.estimator_abstract import EstimatorAbstract
from yaocptool.methods.base.solutionmethodsbase import SolutionMethodsBase
from yaocptool.mpc.plant import Plant, PlantSimulation


class MPC:
    def __init__(self, plant, solution_method, **kwargs):
        """Model Predictive control class. Requires a plant and a solution_method.

        :param Plant|PlantSimulation plant:
        :param SolutionMethodsBase solution_method:
        :param EstimatorAbstract estimator:
        :param DM default_p: is a default parameter vector that will be used by the 'solution_method'
        :param DM default_theta: is a default theta parameter that will be used by the 'solution_method'
        :param bool include_cost_in_state_vector: Typically the optimal control problem has one extra state than the
            plant, the dynamic cost state. By setting this variable to True, it automatically include an zero
            the in state vector obtained from the estimator.
        :param bool mean_as_parameter: The mean estimated by the Estimator as parameter of the OCP. It will be put in
            the end of parameter vector, before the covariance if covariance_as_parameter=True.
        :param bool covariance_as_parameter: The covariance estimated by the Estimator as parameter of the OCP. It will
            be put in the end of parameter vector.
        :param list mean_p_indices: If mean_as_parameter=True, mean_p_indices is a list of tuples (pairs), where the
            first element of the tuple is the index in the mean vector and the second is the index in the p vector.
        :param list cov_p_indices: If covariance_as_parameter=True, cov_p_indices is a list of tuples (pairs), where the
            first element of the tuple is the index in the vectorized covariance matrix and the second is the index
            in the p vector.
        :param function state_rearrangement_function: A function that can be used to rearrange the initial condition in
            cases where the estimated states is not equal to initial condition vector of the OCP. For instance, when the
            OCP has multiple representations of the system. The provided function has the estimated stated as input and
            has to return a initial condition vector.
        """
        self.plant = plant
        self.solution_method = solution_method
        self.estimator = None

        self.last_solutions = None
        self.solution_method_initial_guess = None

        self.statistics = {'iteration_time': []}
        self.n_x = None

        self.iteration = 0

        # Options
        self.p = None
        self.theta = None
        self.verbosity = 1

        self.include_cost_in_state_vector = False
        self.mean_as_parameter = False
        self.mean_p_indices = []
        self.covariance_as_parameter = False
        self.cov_p_indices = []

        self.state_rearrangement_function = None

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        # set last_control_as_parameter to True, so the optimization problem can take last_u
        self.solution_method.last_control_as_parameter = True

    def get_measurement(self):
        """Get measurements from the plant. It will return a tuple with the current measurement and the current control

        :rtype: tuple
        """

        return self.plant.get_measurement()

    def get_states(self, t_k, y_k, u_k):
        """Get states out of a measurement.

        :param DM t_k: time of the measurement
        :param DM y_k: measurements
        :param DM u_k: controls
        :return: DM
        """
        if self.estimator is None:
            x_k = y_k[:self.solution_method.model.n_x]
            cov_x_k = DM.zeros(x_k.shape)
        else:
            x_k, cov_x_k = self.estimator.estimate(t_k, y_k, u_k)

        if self.include_cost_in_state_vector:
            x_k = vertcat(x_k, 0)

        return x_k, cov_x_k

    def get_new_control(self, x_k, u_k, p=None):
        """Use solution_method to obtain new controls

        :param x_k: DM
        :param u_k: DM
        :param p:
        """
        start_time = time.time()
        initial_guess_dict = None
        if self.last_solutions is not None:
            initial_guess_dict = [
                solution.raw_solution_dict for solution in self.last_solutions
            ]
            if len(initial_guess_dict) == 1:
                initial_guess_dict = initial_guess_dict[0]

        solutions = self.solution_method.solve(
            x_0=x_k, initial_guess_dict=initial_guess_dict, p=p, last_u=u_k)

        if not isinstance(solutions, list):
            solutions = [solutions]
        self.last_solutions = solutions

        control = [solution.first_control() for solution in solutions]
        self.statistics['iteration_time'].append(time.time() - start_time)
        return vertcat(*control)

    def send_control(self, u):
        """Sent controls to the plant.

        :param u: DM
        """
        self.plant.set_control(u)

    def post_process_measurement(self, t, meas, u):
        return t, meas, u

    def post_process_states(self, x_k, cov_x_k):
        return x_k, cov_x_k

    def post_process_get_new_control(self, u):
        return u

    def post_process_send_control(self):
        return

    def run(self, iterations=0):
        """Starts computing control and sending it to the plant.

        :param int iterations: the number of iterations that the MPC will run. To run it indefinitely use iterations = 0
        """

        for k in itertools.count(0):
            self.iteration += 1
            if self.verbosity >= 1:
                print(' Iteration {} ({}) '.format(k, self.iteration).center(
                    30, '='))

            # get new measurement from the plant
            t_k, meas_k, u_k = self.get_measurement()
            # post process measurements
            t_k, meas_k, u_k = self.post_process_measurement(t_k, meas_k, u_k)

            if self.verbosity >= 1:
                print('Time: {}'.format(t_k))

            # estimate the states out of the measurement
            x_k, p_k = self.get_states(t_k, meas_k, u_k)
            # post process estimator result
            x_k, p_k = self.post_process_states(x_k, p_k)

            if self.verbosity >= 1:
                print('Estimated state: {}'.format(x_k))

            x_k_ocp = x_k
            if self.state_rearrangement_function is not None:
                x_k_ocp = self.state_rearrangement_function(x_k)

            # get new control using the solution_method
            time_start_new_control = time.time()
            p = self._get_parameter(x_k, p_k)
            new_u = self.get_new_control(x_k=x_k_ocp, u_k=u_k, p=p)
            new_u = self.post_process_get_new_control(new_u)
            if self.verbosity >= 1:
                print('Control calculated: {}'.format(new_u))
                print('Time taken to obtain control: {}'.format(
                    time.time() - time_start_new_control))
            self.send_control(new_u)
            new_u = self.send_control(new_u)

            if k >= iterations - 1:
                print('End of MPC.un'.center(30, '='))
                break

    def _get_parameter(self, x_k, p_k):
        p = self.p

        if self.mean_as_parameter:
            for mean_index, p_index in self.mean_p_indices:
                p[p_index] = x_k[mean_index]
        if self.covariance_as_parameter:
            for cov_index, p_index in self.cov_p_indices:
                p[p_index] = p_k[cov_index]

        return p

    def run_fixed_control(self, u, iterations):
        """
        Run the plant with a fixed control, can be used for initialization purposes.

        :param list|DM|float|int u: control value
        :param int iterations: the number of iterations that the MPC will run.
        """
        if isinstance(u, list):
            u = vertcat(u)

        self.send_control(u)

        for k in range(iterations):
            self.iteration += 1

            if self.verbosity >= 1:
                print(' Iteration {} ({}) '.format(k, self.iteration).center(
                    30, '='))

            # get new measurement from the plant
            t_k, y_k, u_k = self.get_measurement()

            if self.verbosity:
                print('Measurement: {}'.format(y_k))

    def run_fixed_control_with_estimator(self, u, iterations):
        """
        Run the plant with a fixed control, can be used for initialization purposes.

        :param list|DM|float|int u: control value
        :param int iterations: the number of iterations that the MPC will run.
        """
        if isinstance(u, list):
            u = vertcat(u)

        self.send_control(u)

        for k in range(iterations):
            self.iteration += 1

            if self.verbosity:
                print(' Iteration {} ({}) '.format(k, self.iteration).center(
                    30, '='))

            # get new measurement from the plant
            t_k, y_k, u_k = self.get_measurement()

            # estimate the states out of the measurement
            x_k, p_k = self.get_states(t_k, y_k, u_k)
            if self.verbosity:
                print('Measurement: {}'.format(y_k))
                print('Estimated state: {}'.format(x_k))
