import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea

class ErrorPlotter:
    def __init__(self, plots: list, error_window_size: int, error_units_: str, time_units_: str, secondary_axes: bool=False):
        '''
        Class to be used for plotting errors. Default settings will plot some provided error(s) vs. time, although a different
        variable can be plotted along the x-axis by providing the data in the first argument of the update_plot() function
        :param plots: List of plots to be graphed. Names of error plots provided in list will be the default main y-axes labels,
        with secondary y-axes tracking the percent error
        :param error_window_size: Number of prior timesteps to be used to calculate the mean error
        :param error_units_: Units for measuring absolute error (m, cm, rad, etc.)
        :param time_units_: Units of time to be plotted along the x-axis if plotting error vs. time
        :param secondary_axes: Show secondary axis of percent error on plots
        '''
        self.state = None
        self.num_plots = len(plots)
        self.fig, self.axs = plt.subplots(self.num_plots, 1)
        self.lines = []
        self.times = []
        self.twins = []
        self.twin_lines = []
        self.error_data = {}
        self.perc_error_data = {}
        self.window_size = error_window_size
        self.error_units = error_units_
        self.time_units = time_units_
        self.error_window = None
        self.perc_error_window = None
        self.annotation_boxes = []
        self.annotations = []
        self.second_axis = secondary_axes
        idx = 0
        for ax in self.axs:
            ax.set_ylabel(plots[idx] + " (" + self.error_units + ")")
            if self.second_axis:
                twin = ax.twinx()
                self.twins.append(twin)
                twin.set_ylabel(str(plots[idx] + " (Percent)"))
            idx += 1
        self.axs[-1].set_xlabel("Time (" + self.time_units + ")")
        plt.ion()

        self.init = False

    def set_title(self, title):
        self.axs[0].set_title(title)

    def set_xlabel(self, label):
        self.axs[-1].set_xlabel(label)

    def set_main_ylabels(self, *labels):
        idx = 0
        for ax in self.axs:
            ax.set_ylabel(labels[idx])
            idx += 1

    def set_secondary_ylabels(self, *labels):
        if not self.second_axis:
            return
        idx = 0
        for twin in self.twins:
            twin.set_ylabel(labels[idx])
            idx += 1

    def get_average_errors(self):
        error = np.mean(self.error_window, axis=1)
        perc_error = None
        if self.second_axis:
            perc_error = np.mean(self.perc_error_window, axis=1)
        return [error, perc_error]

    def update_plot(self, time: float, *in_data: float):
        '''
        :param time: Timestep associated with updated data if plotting error vs. time, OR other independent variable (i.e. range) to plot error against
        :param data: Data to be plotted. If plotting secondary axis of percent error, use form (error_i, percent_error_i, ...) for n plots
        '''
        plt.ion()
        self.times.append(time)
        if self.second_axis:
            data = in_data[0:-1:2]
            perc_data = in_data[1::2]
        else:
            data = in_data
            perc_data = None
        if not self.init:
            self.error_window = np.array(data).reshape(-1, 1)
            self.perc_error_window = None
            if self.second_axis:
                self.perc_error_window = np.array(perc_data).reshape(-1, 1)
            ave_errors = self.get_average_errors()
            for idx in range(0, len(data)):
                self.error_data[idx] = [data[idx]]
                line, = self.axs[idx].plot(time, data[idx], "r-")
                self.lines.append(line)

                ave_error = ave_errors[0][idx]

                self.annotations.append([TextArea("Absolute Error (" + str(self.window_size) + " window): " + str("{:.3f} ".format(ave_error)) + self.error_units)])
                ab = AnnotationBbox(self.annotations[idx][0], (0.01, 0.9), xycoords='axes fraction', alpha=1.0, pad=0.1, box_alignment=(0, 0))
                self.axs[idx].add_artist(ab)

                if self.second_axis:
                    self.axs[idx].tick_params(axis="y", colors=line.get_color())
                    self.axs[idx].yaxis.label.set_color(line.get_color())

                    self.perc_error_data[idx] = [perc_data[idx]]
                    twin_line, = self.twins[idx].plot(time, perc_data[idx], "b-", zorder=1)
                    self.twin_lines.append(twin_line)
                    self.twins[idx].tick_params(axis="y", colors=twin_line.get_color())
                    self.twins[idx].yaxis.label.set_color(twin_line.get_color())

                    self.axs[idx].set_zorder(self.twins[idx].get_zorder()+1)
                    self.axs[idx].patch.set_visible(False)

                    ave_perc_error = ave_errors[1][idx]

                    self.annotations[idx].append(TextArea("Percent Error (" + str(self.window_size) + " window): " + str("{:.3f}%".format(ave_perc_error))))
                    ab1 = AnnotationBbox(self.annotations[idx][1], (0.01, 0.8), xycoords='axes fraction', alpha=1.0, pad=0.1, box_alignment=(0, 0))
                    self.axs[idx].add_artist(ab1)

            self.init = True
            return

        #  Check if window(s) is/are at maximum size, delete oldest points if needed
        if self.error_window.shape[1] == self.window_size:
            self.error_window = np.delete(self.error_window, 0, 1)
            if self.second_axis:
                self.perc_error_window = np.delete(self.perc_error_window, 0, 1)
        self.error_window = np.append(self.error_window, np.array(data).reshape(-1, 1), axis=1)
        if self.second_axis:
            self.perc_error_window = np.append(self.perc_error_window, np.array(perc_data).reshape(-1, 1), axis=1)
        for idx in range(0, len(data)):
            ave_errors = self.get_average_errors()
            self.error_data[idx].append(data[idx])
            self.lines[idx].set_data(self.times, self.error_data[idx])

            ave_error = ave_errors[0][idx]
            self.annotations[idx][0].set_text("Absolute Error (" + str(self.window_size) + " window): " + str("{:.3f} ".format(ave_error)) + self.error_units)

            self.axs[idx].relim()
            self.axs[idx].autoscale_view(True, True, True)

            if self.second_axis:
                self.perc_error_data[idx].append(perc_data[idx])
                self.twin_lines[idx].set_data(self.times, self.perc_error_data[idx])

                ave_perc_error = ave_errors[1][idx]
                self.annotations[idx][1].set_text("Percent Error (" + str(self.window_size) + " window): " + str("{:.3f}%".format(ave_perc_error)))

                self.twins[idx].relim()
                #self.twins[idx].set_ylim(0, 100)
                self.twins[idx].autoscale_view(True, True, True)

        #plt.show()
        plt.pause(0.0000001)


class PosePlotter:
    def __init__(self, plots: [list], units: str, time_units: str, use_estimates: bool=True):
        '''
        :param plots: List of variable lists to plot on each axis. If a single variable is to be graphed it will be plotted vs time
        :param units: Measurement units of plotted data (used for axis labeling)
        :param time_units: Units of time to be plotted along the x-axis
        '''
        self.num_plots = len(plots)
        self.fig, self.axs = plt.subplots(1, self.num_plots)
        self.units = units
        self.time_units = time_units
        self.times = []
        self.data_lines = []
        self.est_lines = []
        self.data = {}
        self.est_data = {}
        self.plots = plots
        self.use_estimates = use_estimates

        idx = 0
        for ax in self.axs:
            if len(plots[idx]) == 1:
                ax.set_ylabel(plots[idx][0] + " (" + self.units + ")")
                ax.set_xlabel("Time (" + self.time_units + ")")
            elif len(plots[idx]) == 2:
                ax.set_xlabel(plots[idx][0] + " (" + self.units + ")")
                ax.set_ylabel(plots[idx][1] + " (" + self.units + ")")
            else:
                pass  # Does not handle plotting three dimensions
            idx += 1
        plt.ion()
        self.init = False

    def update_plot(self, time: float, *in_data: float):
        '''
        :param time: Timestep associated with updated data
        :param data: Data to be plotted, matching order of variables provided to class constructor, in form (data_i, est_data_i, ...)
        '''
        plt.ion()
        self.times.append(time)
        if self.use_estimates:
            data = in_data[0:-1:2]
            est_data = in_data[1::2]
        else:
            data = in_data
            est_data = None
        if not self.init:
            for d in range(len(data)):
                self.data[d] = [data[d]]
                if self.use_estimates:
                    self.est_data[d] = [est_data[d]]
            data_idx = 0
            for p in range(self.num_plots):
                if len(self.plots[p]) == 1:
                    data_line, = self.axs[p].plot(self.times, self.data[data_idx], "b-")
                    self.data_lines.append(data_line)
                    if self.use_estimates:
                        est_line, = self.axs[p].plot(self.times, self.est_data[data_idx], "r-")
                        self.est_lines.append(est_line)
                        self.axs[p].legend([self.data_lines[p], self.est_lines[p]], ["Actual " + self.plots[p][0], "Estimated " + self.plots[p][0]])
                    data_idx += 1
                elif len(self.plots[p]) == 2:
                    data_line, = self.axs[p].plot(self.data[data_idx], self.data[data_idx + 1], "b-")
                    self.data_lines.append(data_line)
                    if self.use_estimates:
                        est_line, = self.axs[p].plot(self.est_data[data_idx], self.est_data[data_idx + 1], "r-")
                        self.est_lines.append(est_line)
                        self.axs[p].legend([self.data_lines[p], self.est_lines[p]], ["Actual " + self.plots[p][0] + ", " + self.plots[p][1],
                                                                                     "Estimated " + self.plots[p][1] + ", " + self.plots[p][1]])
                    data_idx += 2
                else:
                    pass  # No 3D plotting implemented

            self.init = True

        else:
            for d in range(len(data)):
                self.data[d].append(data[d])
                if self.use_estimates:
                    self.est_data[d].append(est_data[d])
            data_idx = 0
            for p in range(self.num_plots):
                if len(self.plots[p]) == 1:
                    self.data_lines[p].set_data(self.times, self.data[data_idx])
                    if self.use_estimates:
                        self.est_lines[p].set_data(self.times, self.est_data[data_idx])
                    data_idx += 1
                elif len(self.plots[p]) == 2:
                    self.data_lines[p].set_data(self.data[data_idx], self.data[data_idx + 1])
                    if self.use_estimates:
                        self.est_lines[p].set_data(self.est_data[data_idx], self.est_data[data_idx + 1])
                    data_idx += 2

                self.axs[p].relim()
                self.axs[p].autoscale_view(True, True, True)

        plt.pause(0.00001)

    def set_xlabel(self, plot_idx, label):
        self.axs[plot_idx].set_xlabel(label)

    def set_ylabel(self, plot_idx, label):
        self.axs[plot_idx].set_ylabel(label)
