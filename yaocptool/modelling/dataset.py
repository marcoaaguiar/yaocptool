import copy
from collections import defaultdict
from functools import partial
from casadi import horzcat, vertcat, DM

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Failed to import matplotlib. Make sure that it is properly installed')
    plt = None


class DataSet:
    def __init__(self, name='dataset', **kwargs):
        """
            Generic time dependent data storage.

        :param str name:
        :param str plot_style: default plot style. plot = linear interpolation, step = piecewise constant
        ('plot' | 'step')
        """
        self.name = name
        self.data = defaultdict(partial(dict, [('time', DM([])), ('names', None), ('values', DM([])), ('size', None)]))
        self.plot_style = 'step'

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def create_entry(self, entry, size, names=None):
        if names is None:
            names = [entry + '_' + str(i) for i in range(size)]
        self.data[entry]['size'] = size
        self.data[entry]['names'] = names

    def insert_data(self, entry, value, time):
        value = vertcat(value)

        # TODO: Here is a correction for a BUG on CASADI horzcat with DM([]), refactor when CASADI corrects this
        if self.data[entry]['values'].numel() > 0:
            self.data[entry]['values'] = horzcat(self.data[entry]['values'], value)
            self.data[entry]['time'] = horzcat(self.data[entry]['time'], time)
        else:
            self.data[entry]['values'] = value
            self.data[entry]['time'] = time

        if self.data[entry]['size'] is None:
            self.data['entry']['size'] = value.size1()

    def get_copy(self):
        dataset_copy = copy.copy(self)
        dataset_copy.data = copy.deepcopy(self.data)

        return dataset_copy

    def sort(self, entries=None):
        if entries is None:
            entries = self.data.keys()

        for entry in entries:
            time = [self.data[entry]['time'][i] for i in range(self.data[entry]['time'].shape[1])]
            values = [self.data[entry]['values'][:,i] for i in range(self.data[entry]['values'].shape[1])]
            time, values = (list(t) for t in zip(*sorted(zip(time, values), key=lambda point: point[0])))
            self.data[entry]['time'] = horzcat(*time)
            self.data[entry]['values'] = horzcat(*values)


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
