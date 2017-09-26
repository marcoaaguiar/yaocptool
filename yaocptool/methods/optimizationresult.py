import matplotlib.pyplot as plt
from casadi import horzcat, DM
from typing import List


class OptimizationResult:
    def __init__(self, **kwargs):
        # solution_method:
        self.raw_solution_dict = {}
        self.raw_decision_variables = None  # type: List[DM]
        self.finite_elements = -1
        self.degree = -1
        self.degree_control = -1

        self.t_0 = -1
        self.t_f = -1
        self.discretization_scheme = ''

        self.time_breakpoints = []
        self.collocation_points = []

        self.objective = None  # type: DM
        self.constraints_values = None  # type: DM
        self.x_breakpoints_data = {'values': [], 'time': []}
        self.y_breakpoints_data = {'values': [], 'time': []}
        self.u_breakpoints_data = {'values': [], 'time': []}
        self.x_interpolation_data = {'values': [], 'time': []}
        self.y_interpolation_data = {'values': [], 'time': []}
        self.u_interpolation_data = {'values': [], 'time': []}

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    @property
    def is_collocation(self):
        if self.discretization_scheme == '':
            raise Exception('discretization_scheme not defined yet')
        return self.discretization_scheme == 'collocation'

    @property
    def is_valid(self):
        for attr in ['finite_elements', 'degree', 'degree_control', 't_0', 't_f', 'time_breakpoints']:
            if getattr(self, attr) < 0:
                raise Exception('{} attribute {} is lower than 0'.format(self.__class__.__name__, attr))
        for attr in ['objective']:
            if getattr(self, attr) is None:
                raise Exception('{} attribute {} is None'.format(self.__class__.__name__, attr))
        return True

    # Plot
    def plot(self, plot_list, interpolation_data=None):
        if self.is_valid:
            if self.is_collocation and interpolation_data is not False:
                self._plot_interpolation(plot_list)
            else:
                self._plot_breakpoints(plot_list)

    def _plot_breakpoints(self, plot_list):
        x_values = self.x_breakpoints_data['values']
        y_values = self.y_breakpoints_data['values']
        u_values = self.u_breakpoints_data['values']

        t_x = self.x_breakpoints_data['time']
        t_y = self.y_breakpoints_data['time']
        t_u = self.u_breakpoints_data['time']

        for k, entry in enumerate(plot_list):
            fig = plt.figure(k)
            if 'x' in entry:
                for i in entry['x']:
                    plt.plot(t_x, horzcat(*x_values)[i, :].T)
                plt.legend(['x[' + repr(i) + ']' for i in entry['x']])
            if 'y' in entry:
                for i in entry['y']:
                    plt.plot(t_y, horzcat(*y_values)[i, :].T)
                plt.legend(['y[' + repr(i) + ']' for i in entry['y']])

            if 'u' in entry:
                for i in entry['u']:
                    plt.step(t_u, horzcat(*u_values)[i, :].T, where='post')
                plt.legend(['u[' + repr(i) + ']' for i in entry['u']])
            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            k += 1
        plt.show()

    def _plot_interpolation(self, plot_list):
        x_values = self.x_interpolation_data['values']
        y_values = self.y_interpolation_data['values']
        u_values = self.u_interpolation_data['values']

        x_values = horzcat(*[horzcat(*x_values[i]) for i in range(self.finite_elements)])
        y_values = horzcat(*[horzcat(*y_values[i]) for i in range(self.finite_elements)])

        t_x = self.x_interpolation_data['time']
        t_y = self.y_interpolation_data['time']
        t_u = self.u_interpolation_data['time']

        t_x = horzcat(*[horzcat(*t_x[i]) for i in range(self.finite_elements)])
        t_y = horzcat(*[horzcat(*t_y[i]) for i in range(self.finite_elements)])
        t_u = horzcat(*[horzcat(*t_u[i]) for i in range(self.finite_elements)])

        for k, entry in enumerate(plot_list):
            fig = plt.figure(k)

            if 'x' in entry:
                for i in entry['x']:
                    plt.plot(t_x.T, x_values[i, :].T)
                plt.legend(['x[' + repr(i) + ']' for i in entry['x']])

            if 'y' in entry:
                for i in entry['y']:
                    plt.plot(t_y.T, y_values[i, :].T)
                plt.legend(['y[' + repr(i) + ']' for i in entry['y']])

            if 'u' in entry:
                u_value_concat = horzcat(*[horzcat(*u_values[i]) for i in range(self.finite_elements)])

                for i in entry['u']:
                    plt.step(t_u.T, u_value_concat[i, :].T, where='post')
                plt.legend(['u[' + repr(i) + ']' for i in entry['u']])

            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            k += 1
        plt.show()
