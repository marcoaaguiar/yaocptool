import matplotlib.pyplot as plt
from casadi import horzcat


class SimulationResult:
    def __init__(self, **kwargs):
        self.x_names = []
        self.y_names = []
        self.u_names = []
        self.model_name = ''
        self.delta_t = 1
        self.t_0 = 0
        self.t_f = 1
        self.finite_elements = 0
        self.n_x = 0
        self.n_y = 0
        self.n_u = 0
        self.x = []
        self.y = []
        self.u = []
        self.t = []

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def final_condition(self):
        """Return the simulation final condition in the form of a tuple (x_f, y_f, u_f)

        :rtype: DM, DM, DM
        """
        return self.x[-1][-1], self.y[-1][-1], self.u[-1][-1]

    def plot(self, plot_list):
        """Plot the simulation results.
        It takes as input a list of dictionaries, each dictionary represents a plot.  In the dictionary use keyword 'x'
        to specify which states you want to print, the value of the dictionary should be a list of state to be printed.
        The keywords that are accepted are: 'x', 'y', 'u'
        :param list plot_list: List of dictionaries to generate the plots.
        """

        if isinstance(plot_list, dict):
            plot_list = [plot_list]

        x_values = self.x
        y_values = self.y
        u_values = self.u

        x_values = horzcat(*[horzcat(*x_values[l]) for l in range(self.finite_elements + 1)])
        y_values = horzcat(*[horzcat(*y_values[l]) for l in range(self.finite_elements)])
        u_values = horzcat(*[horzcat(*u_values[l]) for l in range(self.finite_elements)])

        for k, entry in enumerate(plot_list):
            fig = plt.figure(k)
            lines = []
            # Plot optimization x data
            if 'x' in entry:
                for l in entry['x']:
                    line = self._plot_entry(self.t, x_values, l, label=self.x_names[l], plot_style='plot')
                    lines.append(line)

            # Plot optimization y data
            if 'y' in entry:
                for l in entry['y']:
                    line = self._plot_entry(self.t[1:], y_values, l, label=self.y_names[l], plot_style='plot')
                    lines.append(line)

            # Plot optimization u data
            if 'u' in entry:
                for l in entry['u']:
                    plot_style = 'step'
                    line = self._plot_entry(self.t[:-1], u_values, l, label=self.u_names[l], plot_style=plot_style)
                    lines.append(line)

            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            plt.legend()
        plt.show()

    @staticmethod
    def _plot_entry(t_vector, data_vector, row, label='', plot_style='plot'):
        if plot_style not in ['plot', 'step']:
            raise ValueError('Plot style not recognized: "{}". Allowed : "plot" and "step"'.format(plot_style))
        if plot_style == 'plot':
            return plt.plot(t_vector, data_vector[row, :].T, label=label)
        elif plot_style == 'step':
            return plt.step(t_vector, data_vector[row, :].T, label=label, where='post')
