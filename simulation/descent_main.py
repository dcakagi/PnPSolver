import airsim
import numpy as np
import sys
import time
import cv2
import os
sys.path.append("../")

from PoseEstimator import PoseEstimator
from Plotter import ErrorPlotter
from light_detector import LightDetector


client = airsim.MultirotorClient("10.32.114.170")
client.confirmConnection()

landing_point = np.array([-35880.0, -41940.0, 4870.0])
start_location = np.array([-56858.316406, -43683.730469, 5117.776855])

constellation_2 = {"cyan_blue": [-305, 305, 0], "purple": [305, 305, 0],
                   "light_magenta": [305, -305, 0], "yellow_green": [-305, -305, 0]}

# constellation_file = open("data_300m_zoom/constellation_2.pkl","wb")
# pickle.dump(constellation_2,constellation_file)
# constellation_file.close()

pix_height = 2160
pix_width = 3840
pix_diag = np.sqrt(pix_height**2 + pix_width**2)
down_fov_diag = 5 * np.pi / 6  # Fisheye lens for downward facing camera - 150 deg FOV (diagonal)

focal_pix = pix_diag / (2 * np.tan(down_fov_diag / 2))

estimator = PoseEstimator(constellation_2, pix_width, pix_height, focal_pix, solver_type='opencv', method=cv2.SOLVEPNP_IPPE_SQUARE, use_guess=True)

light_detector = LightDetector()

error_plots = ["Horiz", "Vert"]
error_plot = ErrorPlotter(error_plots, 10, 'm', 's', secondary_axes=False)
error_plot.set_xlabel("Range (m)")
error_plot.set_main_ylabels("Horizontal Error (m)", "Vertical Error (m)")

idx = 0
while True:
    start_time = time.time()

    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])

    position = responses[0].camera_position
    orientation = responses[0].camera_orientation
    image_bytes = responses[0].image_data_uint8

    position.x_val *= 100
    position.y_val *= 100
    position.z_val *= 100

    position.x_val += start_location[0]
    position.y_val += start_location[1]
    position.z_val += start_location[2]

    position.x_val -= landing_point[0]
    position.y_val -= landing_point[1]
    position.z_val -= landing_point[2]

    image_1d = np.frombuffer(image_bytes, dtype=np.uint8)
    image_rgb = image_1d.reshape(responses[0].height, responses[0].width, 3)  # Height x width? Pixels given in width x height?

    # write images and other data to file
    # cv2.imwrite(os.path.normpath("../data_60m_descent/pictures/pic_" + str(idx) + '.png'), image_rgb)
    idx += 1

    pixel_locations = light_detector.final_detect(image_rgb, "front")  # Down? Name isn't used
    estimated_camera_location = estimator.updatePose(pixel_locations)

    x_est_cv = estimated_camera_location[0, 0]
    y_est_cv = estimated_camera_location[1, 0]
    z_est_cv = estimated_camera_location[2, 0]

    x_val = position.x_val
    y_val = position.y_val
    z_val = position.z_val

    x_error = x_est_cv - x_val
    y_error = y_est_cv - y_val
    z_error = z_est_cv - z_val

    range = np.linalg.norm([x_val, y_val, z_val])
    horiz_error = np.linalg.norm([x_error, y_error])
    vert_error = np.linalg.norm([z_error])

    error_plot.update_plot(range, horiz_error/100, vert_error/100)
    print("secs: " + str(time.time()-start_time))

