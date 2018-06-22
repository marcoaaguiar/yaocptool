from collections import defaultdict
from functools import partial

try:
    import matplotlib.pyplot as plt
except:
    print('Failed to import matplotlib. Make sure that is properly installed')
from casadi import horzcat, vertcat


class DataSet:
    def __init__(self, name='dataset', **kwargs):
        """
            Generic time dependent data storage.

        :param str name:
        :param str plot_style: default plot style. plot = linear interpolation, step = piecewise constant
        ('plot' | 'step')
        """
        self.name = name
        self.data = defaultdict(partial(dict, [('time', []), ('names', None), ('values', []), ('size', None)]))
        self.plot_style = 'step'

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def insert_data(self, entry, value, time):
        value = vertcat(value)

        self.data[entry]['values'] = horzcat(self.data[entry]['values'], value)
        self.data[entry]['time'] = horzcat(self.data[entry]['time'], time)

        if self.data[entry]['size'] is None:
            self.data['entry']['size'] = value.size1()

    @staticmethod
    def _plot_entry(t_vector, data_vector, row, label='', plot_style='plot'):
        if plot_style not in ['plot', 'step']:
            raise ValueError('Plot style not recognized: "{}". Allowed : "plot" and "step"'.format(plot_style))
        if plot_style == 'plot':
            return plt.plot(t_vector.T, data_vector[row, :].T, label=label)
        elif plot_style == 'step':
            return plt.step(t_vector.T, data_vector[row, :].T, label=label, where='post')

    def plot(self, plot_list, figures=None, show=True):
        """Plot DataSet information.
        It takes as input a list of dictionaries, each dictionary represents a plot.  In the dictionary use keyword 'x'
        to specify which states you want to print, the value of the dictionary should be a list of index of the states
        to be printed.
        :param list plot_list: List of dictionaries to generate the plots.
        :param list figures: list of figures to be plotted on top (optional)
        :param bool show: if the plotted figures should be shown after plotting (optional, default=True).
        """

        if isinstance(plot_list, dict):
            plot_list = [plot_list]

        used_figures = []

        for k, entry_dict in enumerate(plot_list):
            if figures is not None:
                fig = plt.figure(figures[k].number)
            else:
                fig = plt.figure(k)

            used_figures.append(fig)

            lines = []
            # Plot optimization x data
            for key in entry_dict:
                if entry_dict[key] == 'all':
                    entry_dict[key] = range(self.data[key]['size'])
                for l in entry_dict[key]:
                    line = self._plot_entry(self.data[key]['time'], self.data[key]['values'], l,
                                            label=self.data[key]['names'][l], plot_style=self.plot_style)
                    lines.append(line)

            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            plt.legend()
        plt.interactive(True)
        if show:
            plt.show()

        return used_figures
