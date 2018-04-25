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
        :param bool include_cost_in_state_vector: Typically the optimal control problem has one extra state than the
        plant, the dynamic cost state. By setting this variable to True, it automatically include an zero the in state
        vector obtained from the estimator.
        """
        self.plant = plant
        self.solution_method = solution_method
        self.estimator = None

        self.last_solutions = None
        self.solution_method_initial_guess = None

        self.statistics = {'iteration_time': []}
        self.n_x = None

        # Options
        self.include_cost_in_state_vector = False

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if self.n_x is None:
            self.n_x = self.plant.n_x

    def get_new_control(self, x_k, u_k):
        """Use solution_method to obtain new controls

        :param x_k: DM
        :param u_k: DM
        """
        start_time = time.time()
        initial_guess_dict = None
        if self.last_solutions is not None:
            initial_guess_dict = [solution.raw_solution_dict for solution in self.last_solutions]
            if len(initial_guess_dict) == 1:
                initial_guess_dict = initial_guess_dict[0]

        solutions = self.solution_method.solve(x_0=x_k, initial_guess_dict=initial_guess_dict)

        if not isinstance(solutions, list):
            solutions = [solutions]
        self.last_solutions = solutions

        control = [solution.first_control() for solution in solutions]
        self.statistics['iteration_time'].append(time.time() - start_time)
        return vertcat(*control)

    def get_measurement(self):
        """Get measurements from the plant. It will return a tuple with the current measurement and the current control

        :rtype: tuple
        """

        return self.plant.get_measurement()

    def send_control(self, u):
        """Sent controls to the plant.

        :param u: DM
        """
        self.plant.set_control(u)

    def get_states(self, t_k, y_k, u_k):
        """Get states out of a measurement.

        :param DM t_k: time of the measurement
        :param DM y_k: measurements
        :param DM u_k: controls
        :return: DM
        """
        if self.estimator is None:
            x_k = y_k[:self.n_x]
            p_k = DM.zeros(x_k.shape)
        else:
            x_k, p_k = self.estimator.estimate(t_k, y_k, u_k)

        if self.include_cost_in_state_vector:
            x_k = vertcat(x_k, 0)

        return x_k, p_k

    def run(self, iterations=0):
        """Starts computing control and sending it to the plant.

        :param int iterations: the number of iterations that the MPC will run. To run it indefinitely use iterations = 0
        """

        for k in itertools.count(0):
            print(' Iteration {} '.format(k).center(30, '='))
            # get new measurement from the plant
            t_k, y_k, u_k = self.get_measurement()
            print('Time: {}'.format(t_k))

            # estimate the states out of the measurement
            x_k, p_k = self.get_states(t_k, y_k, u_k)
            print('State: {}'.format(x_k))

            # get new control from the
            time_start_new_control = time.time()
            new_u = self.get_new_control(x_k=x_k, u_k=u_k)
            print('Control calculated: {}'.format(new_u))
            print('Time taken to obtain control: {}'.format(time.time() - time_start_new_control))
            self.send_control(new_u)

            if k >= iterations - 1:
                print('End of MPC.un'.center(30, '='))
                break

    def run_fixed_control(self, u, iterations, verbosity=1):
        """
        Run the plant with a fixed control, can be used for initialization purposes.

        :param list|DM|floar|int u: control value
        :param int iterations: the number of iterations that the MPC will run.
        :param verbosity: indicates if it should print information about the iterations
        """
        if isinstance(u, list):
            u = vertcat(u)

        self.send_control(u)

        for k in range(iterations):
            if verbosity:
                print(' Iteration {} '.format(k).center(30, '='))

            # get new measurement from the plant
            t_k, y_k, u_k = self.get_measurement()

            if verbosity:
                print('Measurement: {}'.format(y_k))

    def run_fixed_control_with_estimator(self, u, iterations, verbosity=1):
        """
        Run the plant with a fixed control, can be used for initialization purposes.

        :param list|DM|float|int u: control value
        :param int iterations: the number of iterations that the MPC will run.
        :param verbosity: indicates if it should print information about the iterations
        """
        if isinstance(u, list):
            u = vertcat(u)

        self.send_control(u)

        for k in range(iterations):
            if verbosity:
                print(' Iteration {} '.format(k).center(30, '='))

            # get new measurement from the plant
            t_k, y_k, u_k = self.get_measurement()

            # estimate the states out of the measurement
            x_k = self.get_states(t_k, y_k, u_k)
            if verbosity:
                print('Measurement: {}'.format(y_k))
                print('Estimated state: {}'.format(x_k))
