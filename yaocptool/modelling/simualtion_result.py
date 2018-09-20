try:
    import matplotlib.pyplot as plt
except:
    print('Failed to import matplotlib. Make sure that it is properly installed')
from casadi import horzcat, vertcat


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

    def extend(self, other_sim_result):
        """Extend this SimulationResult with other SimulationResult.
        It is only implemented for the case where the other simulation result starts at the end of this
        simulation. That is this.t_f == other_sim_result.t_0 .

        :param SimulationResult other_sim_result:
        :return:
        """
        list_of_attributes_to_check = ['n_x', 'n_y', 'n_u']
        for attr in list_of_attributes_to_check:
            if not getattr(self, attr) == getattr(other_sim_result, attr):
                raise Exception(
                    "Attribute {} is no equal for both SimulationResults: {}!={}".format(attr, getattr(self, attr),
                                                                                         getattr(other_sim_result,
                                                                                                 attr)))
        if not other_sim_result.t_0 >= self.t_f:
            raise Exception("Merge method only implemented for merging simulation results that are subsequent")

        self.t_f = other_sim_result.t_f
        self.finite_elements += other_sim_result.finite_elements

        self.x.extend(other_sim_result.x[1:])
        self.y.extend(other_sim_result.y)
        self.u.extend(other_sim_result.u)
        self.t = vertcat(self.t, other_sim_result.t[1:])

    def plot(self, plot_list, figures=None, show=True):
        """Plot the simulation results.
        It takes as input a list of dictionaries, each dictionary represents a plot.  In the dictionary use keyword 'x'
        to specify which states you want to print, the value of the dictionary should be a list of state to be printed.
        The keywords that are accepted are: 'x', 'y', 'u'
        :param list plot_list: List of dictionaries to generate the plots.
        :param list figures: list of figures to be plotted on top (optional)
        :param bool show: if the plotted figures should be shown after plotting (optional, default=True).
        """

        if isinstance(plot_list, dict):
            plot_list = [plot_list]

        used_figures = []

        x_values = self.x
        y_values = self.y
        u_values = self.u

        x_values = horzcat(*[horzcat(*x_values[l]) for l in range(self.finite_elements + 1)])
        y_values = horzcat(*[horzcat(*y_values[l]) for l in range(self.finite_elements)])
        u_values = horzcat(*[horzcat(*u_values[l]) for l in range(self.finite_elements)])

        for k, entry in enumerate(plot_list):
            if figures is not None:
                fig = plt.figure(figures[k].number)
            else:
                fig = plt.figure(k)

            used_figures.append(fig)

            lines = []
            # Plot optimization x data
            if 'x' in entry:
                x_indices = entry['x']
                if x_indices == 'all':
                    x_indices = range(x_values.shape[0])
                for l in x_indices:
                    line = self._plot_entry(self.t, x_values, l, label=self.x_names[l], plot_style='plot')
                    lines.append(line)

            # Plot optimization y data
            if 'y' in entry:
                y_indices = entry['y']
                if y_indices == 'all':
                    y_indices = range(y_values.shape[0])
                for l in y_indices:
                    line = self._plot_entry(self.t[1:], y_values, l, label=self.y_names[l], plot_style='plot')
                    lines.append(line)

            # Plot optimization u data
            if 'u' in entry:
                u_indices = entry['u']
                if u_indices == 'all':
                    u_indices = range(u_values.shape[0])
                for l in u_indices:
                    plot_style = 'step'
                    line = self._plot_entry(self.t[:-1], u_values, l, label=self.u_names[l], plot_style=plot_style)
                    lines.append(line)

            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            plt.legend()
        plt.interactive(True)
        if show:
            plt.show()

        return used_figures

    @staticmethod
    def _plot_entry(t_vector, data_vector, row, label='', plot_style='plot'):
        if plot_style not in ['plot', 'step']:
            raise ValueError('Plot style not recognized: "{}". Allowed : "plot" and "step"'.format(plot_style))
        if plot_style == 'plot':
            return plt.plot(t_vector, data_vector[row, :].T, label=label)
        elif plot_style == 'step':
            return plt.step(t_vector, data_vector[row, :].T, label=label, where='post')
