import airsim
import numpy as np
import os
import cv2
import time
import sys
sys.path.append("../")

from PoseEstimator import PoseEstimator
from Plotter import ErrorPlotter
from Plotter import PosePlotter
from light_detector import LightDetector
import pickle

# connect to the AirSim simulator
client = airsim.MultirotorClient("10.32.114.100")
client.confirmConnection()

landing_point = np.array([-35880.0,-41940.0,4870.0])
start_location = np.array([-56858.316406,-43683.730469,5117.776855])

constellation = {"cyan": [818.195312, -971.097656, 458.900879], "magenta": [867.039062, -11.128906, 674.893066],
                     "yellow": [867.039062, 698.871094, 524.893066], "red": [767.039062, 308.871094, 164.893066],
                     "green": [17.039062, -631.128906, 634.893066], "blue": [231.492188, 770.839844, 155.332031],
                     "red_magenta": [-664.863281, -416.613281, 423.383301], "orange": [-1042.960938, -1351.128906, 964.893066],
                     "blue_green": [-1052.960938, 78.871094, 584.893066]}

# constellation_file = open("data_300m/constellation.pkl","wb")
# pickle.dump(constellation,constellation_file)
# constellation_file.close()

pixels_height = 2160
pixels_width = 3840
fov = np.pi / 2
focal_pixels = pixels_width / (2 * np.tan(fov / 2))

# estimator = PoseEstimator(constellation, pixels_width, pixels_height, focal_pixels)
estimator_cv = PoseEstimator(constellation, pixels_width, pixels_height, focal_pixels, solver_type="opencv")


## initialize error plots
# error_plots = ["X Error", "Y Error", "Z Error"]
# plotter = ErrorPlotter(error_plots, 50, 'm', 's')
# plotter.set_title("Errors")
#Change default x and y axes labels if needed using ErrorPlotter.set_xlabel(), etc.
pose_plots = [["X", "Y"], ["Z"]]  # Plotting x vs y, and z (2 plots)
pose_plot = PosePlotter(pose_plots, 'm', 's')
pose_plot.set_xlabel(0, "X Position (m)")  # Change default axes labels if needed
pose_plot.set_ylabel(0, "Y Position (m)")
pose_plot.set_ylabel(1, "Z Position (m)")

# rotation_plots = ["Pitch", "Roll", "Yaw"]
# rot_plotter = ErrorPlotter(rotation_plots, 100, 'rad', 's')
# rot_plotter.set_title("Rotational Errors")

# video writer
# fourcc = cv2.VideoWriter_fourcc(*'avc1')
# video_out = cv2.VideoWriter('output.mp4',fourcc, 1/.7, (pixels_width,pixels_height))

#start the picture grabing and estimations
light_detector = LightDetector()
start = time.time()
first = True
i = 0

actual_positions = []
while(True):
# while (time.time()-start < 10):
    start_time = time.time()

    responses = client.simGetImages([
                        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
    # file = open("FOV","r")
    # fov = np.deg2rad(float(file.read()))
    # estimator_cv.setFocalLengthPixelsFromFOV(fov)

    if first:
        start_time_stamp = responses[0].time_stamp
        first = False
    position = responses[0].camera_position
    orientation = responses[0].camera_orientation
    image_bytes = responses[0].image_data_uint8

    position.x_val *= 100
    position.y_val *= 100
    position.z_val *= -100

    position.x_val += start_location[0]
    position.y_val += start_location[1]
    position.z_val += start_location[2]

    position.x_val -= landing_point[0]
    position.y_val -= landing_point[1]
    position.z_val -= landing_point[2]
    
    
    # print("camera Position: ")
    # print(position)
    # print()
    # print("camera orientation (quaternion): ")
    # print(orientation)

    image_1d = np.frombuffer(image_bytes, dtype=np.uint8) 

    # reshape array to 3 channel image array H X W X 3
    image_rgb = image_1d.reshape(responses[0].height, responses[0].width, 3)
    
    # # write images and other data to file
    # cv2.imwrite(os.path.normpath("data_300m/pictures/pic_" + str(i) + '.png'), image_rgb)
    
    # actual_positions = np.array([position.x_val, position.y_val,position.z_val])
    # with open('data_300m/actual_positions.txt','ab') as f:
    #     np.savetxt(f, actual_positions.reshape(1,3), delimiter=',')  

    pixel_locations = light_detector.detect(image_rgb,"front")
    # pixel_file = open("data_300m/pixel_locations/px_loc_" + str(i) + ".pkl","wb")
    # pickle.dump(pixel_locations,pixel_file)
    # pixel_file.close()
    # i += 1 

    print(len(pixel_locations.values()))
    if len(pixel_locations.values()) < 6:
        print( "Not enought points detected")
        continue
    # estimated_camera_location = estimator.updatePose(pixel_locations)
    estimated_camera_location_cv = estimator_cv.updatePose(pixel_locations)

    # x_est_cv = estimated_camera_location[0, 0]
    # y_est_cv = estimated_camera_location[1, 0]
    # z_est_cv = estimated_camera_location[2, 0]
    x_est_cv = estimated_camera_location_cv[0, 0]
    y_est_cv = estimated_camera_location_cv[1, 0]
    z_est_cv = estimated_camera_location_cv[2, 0]

    x_val = position.x_val 
    y_val = position.y_val
    z_val = position.z_val

    x_error_cv = x_est_cv - x_val
    y_error_cv = y_est_cv - y_val
    z_error_cv = z_est_cv - z_val

    # print(f"\nOPnP X estimate: {x_est}")
    # print(f"OPnP Y estimate: {y_est}")
    # print(f"OPnP Z estimate: {z_est}")

    # print(f"\nOPnP X % error: {x_error}")
    # print(f"OPnP Y % error: {y_error}")
    # print(f"OPnP Z % error: {z_error}")

    # print(f"\nOpencv X estimate: {x_est_cv}")
    # print(f"Opencv Y estimate: {y_est_cv}")
    # print(f"Opencv Z estimate: {z_est_cv}")

    # print(f"\nOpencv X % error: {x_error_cv}")
    # print(f"Opencv Y % error: {y_error_cv}")
    # print(f"Opencv Z % error: {z_error_cv}")

    t = (responses[0].time_stamp - start_time_stamp) / 1000000000
    x_perc_error_cv = abs(x_error_cv / x_val) * 100
    # y_perc_error_cv = abs((y_error_cv +1 )/ (1+y_val)) * 100
    y_perc_error_cv = abs(np.log(np.exp(y_error_cv)))
    z_perc_error_cv = abs(z_error_cv / z_val) * 100

    # print(f"\nOpencv X % error: {x_perc_error_cv}")
    # print(f"Opencv Y % error: {y_perc_error_cv}")
    # print(f"Opencv Z % error: {z_perc_error_cv}")

    # plotter.update_plot(t, x_error_cv/100, x_perc_error_cv, y_error_cv/100, y_perc_error_cv, z_error_cv/100, z_perc_error_cv)
    pose_plot.update_plot(t, x_val/100, x_est_cv/100, y_val/100, y_est_cv/100, z_val/100, z_est_cv/100)
    # print("updated")
    print("secs: " + str(time.time()-start_time))