import itertools

from casadi import vertcat

from yaocptool.methods.base.solutionmethodsbase import SolutionMethodsBase
from yaocptool.modelling.simualtion_result import SimulationResult
from yaocptool.mpc.plant import Plant, PlantSimulation


class MPC:
    def __init__(self, plant, solution_method, **kwargs):
        """Model Predictive control class. Requires a plant and a solution_method.

        :param plant: Plant or PlantSimulation object
        :param solution_method:
        :param kwargs:
        """
        self.plant = plant
        self.solution_method = solution_method  # type: SolutionMethodsBase

        self.solution_method_initial_guess = None

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def run(self, iterations=0):
        """Starts computing control and sending it to the plant.
        It cna receive two arguments

        :param int iterations: the number of iterations that the MPC will run. To run it indefinitely use iterations = 0
        """

        for k in itertools.count(0):
            print(' Iteration {} '.format(k).center(30, '='))
            # get new measurement from the plant
            t_k, y_k, u_k = self.get_measurement()
            print('Time: {}'.format(t_k))

            # estimate the states out of the measurement
            x_k = self.get_states(y_k)
            print('State: {}'.format(x_k))

            # get new control from the
            new_u = self.get_new_control(x_k=x_k, u_k=u_k)
            print('Control calculated: {}'.format(new_u))

            self.send_control(new_u)

            if k >= iterations:
                print('End of MPC.un'.center(30, '='))
                break

    def get_new_control(self, x_k, u_k):
        """Use solution_method to obtain new controls

        :param x_k: DM
        :param u_k: DM
        """
        solutions = self.solution_method.solve(x_0=x_k, initial_guess_dict=self.solution_method_initial_guess)
        if not isinstance(solutions, list):
            solutions = [solutions]

        self.solution_method_initial_guess = [solution.raw_solution_dict for solution in solutions]
        control = [solution.first_control() for solution in solutions]
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

    def get_states(self, y_k):
        """Get states out of a measurement.

        :param y_k: DM
        :return: DM
        """
        x_k = y_k[self.solution_method.model.n_x:]
        return x_k
