import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea

class ErrorPlotter:
    def __init__(self, plots: list, error_window_size: int, error_units_: str, time_units_: str):
        '''
        :param plots: List of plots to be graphed. Names of error plots provided in list will be the default main y-axes labels,
        with secondary y-axes tracking the percent error
        :param error_window_size: Number of prior timesteps to be used to calculate the mean error
        :param error_units_: Units for measuring absolute error (m, cm, rad, etc.)
        :param time_units_: Units of time to be plotted along the x-axis
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
        idx = 0
        for ax in self.axs:
            twin = ax.twinx()
            self.twins.append(twin)
            ax.set_ylabel(plots[idx] + " (" + self.error_units + ")")
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
        idx = 0
        for twin in self.twins:
            twin.set_ylabel(labels[idx])
            idx += 1

    def get_average_errors(self):
        error = np.mean(self.error_window, axis=1)
        perc_error = np.mean(self.perc_error_window, axis=1)
        return [error, perc_error]

    def update_plot(self, time: float, *data: list):
        '''
        :param time: Timestep associated with updated data
        :param data: List of data in form [error_i, percent_error_i, ...] for n plots
        '''
        plt.ion()
        self.times.append(time)
        if not self.init:
            self.error_window = np.array(data[0:-1:2]).reshape(-1, 1)
            self.perc_error_window = np.array(data[1::2]).reshape(-1, 1)
            ave_errors = self.get_average_errors()
            for idx in range(0, int(len(data)/2)):
                self.error_data[idx] = [data[2*idx]]
                self.perc_error_data[idx] = [data[2*idx + 1]]
                line, = self.axs[idx].plot(time, data[idx], "r-")
                twin_line, = self.twins[idx].plot(time, data[2*idx + 1], "b-", zorder=1)
                self.lines.append(line)
                self.twin_lines.append(twin_line)

                self.axs[idx].tick_params(axis="y", colors=line.get_color())
                self.twins[idx].tick_params(axis="y", colors=twin_line.get_color())
                self.axs[idx].yaxis.label.set_color(line.get_color())
                self.twins[idx].yaxis.label.set_color(twin_line.get_color())

                self.axs[idx].set_zorder(self.twins[idx].get_zorder()+1)
                self.axs[idx].patch.set_visible(False)

                ave_error = ave_errors[0][idx]
                ave_perc_error = ave_errors[1][idx]

                self.annotations.append([TextArea("Absolute Error (" + str(self.window_size) + " window): " + str("{:.3f} ".format(ave_error)) + self.error_units)])
                ab = AnnotationBbox(self.annotations[idx][0], (0.01, 0.9), xycoords='axes fraction', alpha=1.0, pad=0.1, box_alignment=(0, 0))
                self.axs[idx].add_artist(ab)

                self.annotations[idx].append(TextArea("Percent Error (" + str(self.window_size) + " window): " + str("{:.3f}%".format(ave_perc_error))))
                ab1 = AnnotationBbox(self.annotations[idx][1], (0.01, 0.8), xycoords='axes fraction', alpha=1.0, pad=0.1, box_alignment=(0, 0))
                self.axs[idx].add_artist(ab1)

            self.init = True
            return

        if self.error_window.shape[1] == self.window_size:
            self.error_window = np.delete(self.error_window, 0, 1)
            self.perc_error_window = np.delete(self.perc_error_window, 0, 1)
        self.error_window = np.append(self.error_window, np.array(data[0:-1:2]).reshape(-1, 1), axis=1)
        self.perc_error_window = np.append(self.perc_error_window, np.array(data[1::2]).reshape(-1, 1), axis=1)
        for idx in range(0, int(len(data)/2)):
            ave_errors = self.get_average_errors()
            self.error_data[idx].append(data[2*idx])
            self.perc_error_data[idx].append(data[2*idx + 1])
            self.lines[idx].set_data(self.times, self.error_data[idx])
            self.twin_lines[idx].set_data(self.times, self.perc_error_data[idx])

            ave_error = ave_errors[0][idx]
            ave_perc_error = ave_errors[1][idx]

            self.annotations[idx][0].set_text("Absolute Error (" + str(self.window_size) + " window): " + str("{:.3f} ".format(ave_error)) + self.error_units)
            self.annotations[idx][1].set_text("Percent Error (" + str(self.window_size) + " window): " + str("{:.3f}%".format(ave_perc_error)))

            self.axs[idx].relim()
            #self.twins[idx].set_ylim(0, 100)
            self.twins[idx].relim()
            self.axs[idx].autoscale_view(True, True, True)
            self.twins[idx].autoscale_view(True, True, True)
        #plt.show()
        plt.pause(0.0000001)


if __name__ == '__main__':
    error_plots = ["X Error", "Y Error", "Z Error"]
    plotter = ErrorPlotter(error_plots, 50, 'm', 's')
    plotter.set_title("Errors")
    #Change default x and y axes labels if needed using ErrorPlotter.set_xlabel(), etc.

    rotation_plots = ["Pitch", "Roll", "Yaw"]
    rot_plotter = ErrorPlotter(rotation_plots, 100, 'rad', 's')
    rot_plotter.set_title("Rotational Errors")

    for i in range(1000):
        # Make fake data
        t = i
        x_error = np.random.random() / 10
        x_perc_error = x_error * np.random.random() * 100
        y_error = np.random.random() / 10
        y_perc_error = y_error * np.random.random() * 100
        z_error = np.random.random() / 10
        z_perc_error = z_error * np.random.random() * 100

        # Update plots with data (error_i, %error_i,...)
        plotter.update_plot(t, x_error, x_perc_error, y_error, y_perc_error, z_error, z_perc_error)

        pitch_error = np.random.random() * 3
        pitch_perc_error = x_error * np.random.random() * 50
        roll_error = np.random.random() * 3
        roll_perc_error = y_error * np.random.random() * 50
        yaw_error = np.random.random() * 3
        yaw_perc_error = z_error * np.random.random() * 50

        rot_plotter.update_plot(t, pitch_error, pitch_perc_error, roll_error, roll_perc_error, roll_error, roll_perc_error)