from Plotter import *

if __name__ == '__main__':
    error_plots = ["X Error", "Y Error", "Z Error"]
    plotter = ErrorPlotter(error_plots, 50, 'm', 's', secondary_axes=True)
    plotter.set_title("Errors")
    #  Change default x and y axes labels if needed using ErrorPlotter.set_xlabel(), etc.

    rotation_plots = ["Pitch", "Roll", "Yaw"]
    rot_plotter = ErrorPlotter(rotation_plots, 100, 'rad', 's', secondary_axes=False)
    rot_plotter.set_title("Rotational Errors")

    # pose_plots = [["X"], ["Y"], ["Z"]]  # Plotting x, y, and z independently (3 plots)
    pose_plots = [["X", "Y"], ["Z"]]  # Plotting x vs y, and z (2 plots)
    pose_plot = PosePlotter(pose_plots, 'm', 's')
    pose_plot.set_xlabel(0, "X Position (m)")  # Change default axes labels if needed
    pose_plot.set_ylabel(0, "Y Position (m)")
    pose_plot.set_ylabel(1, "Z Position (m)")

    range_plots = ["Horizontal Error", "Vertical Error"]
    range_error_plot = ErrorPlotter(range_plots, 50, 'm', 's')
    range_error_plot.set_xlabel("Range (m)")

    eps = 1e-6  # Avoid divide by zero errors
    for i in range(200):
        # Make fake data
        t = i

        x_pos = 50 * np.sin(t/50) + t + 10
        x_est = x_pos + np.random.random() * 2
        x_error = np.abs(x_est - x_pos)
        x_perc_error = np.abs(x_error / (x_pos + eps)) * 100

        y_pos = 50 * np.cos(t/50) + t + 10
        y_est = y_pos + np.random.random() * 2
        y_error = np.abs(y_pos - y_est)
        y_perc_error = np.abs(y_error / (y_pos + eps)) * 100

        z_pos = 1.5 * t + 5
        z_est = z_pos + np.random.random() * 2
        z_error = np.abs(z_pos - z_est)
        z_perc_error = np.abs(z_error / (z_pos + eps)) * 100

        horizontal_error = np.sqrt(x_error**2 + y_error**2)
        vertical_error = np.sqrt(y_error**2)
        range = np.sqrt(x_pos**2 + y_pos**2 + z_pos**2)

        range_error_plot.update_plot(range, horizontal_error, vertical_error)

        # Update pose plots with data (time, data_i, est_data_i, ...)
        pose_plot.update_plot(t, x_pos, x_est, y_pos, y_est, z_pos, z_est)

        # Update error plots with data (time, error_i, %error_i, ...)
        plotter.update_plot(t, x_error, x_perc_error, y_error, y_perc_error, z_error, z_perc_error)

        # Rotational error plotting
        pitch_error = np.random.random() * 3
        pitch_perc_error = x_error * np.random.random() * 50
        roll_error = np.random.random() * 3
        roll_perc_error = y_error * np.random.random() * 50
        yaw_error = np.random.random() * 3
        yaw_perc_error = z_error * np.random.random() * 50

        #rot_plotter.update_plot(t, pitch_error, pitch_perc_error, roll_error, roll_perc_error, yaw_error, yaw_perc_error)
        rot_plotter.update_plot(t, pitch_error, roll_error, yaw_error)
