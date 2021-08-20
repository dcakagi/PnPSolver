import airsim
import numpy as np
import time
import sys
sys.path.append("../")

from PoseEstimator import PoseEstimator
from Plotter import ErrorPlotter
from Plotter import PosePlotter
from light_detector import LightDetector

import pickle
import cv2
import os

# connect to the AirSim simulator
client = airsim.MultirotorClient("10.32.114.184")
# client = airsim.MultirotorClient()
client.confirmConnection()

landing_point = np.array([-35880.0,-41940.0,4870.0])
start_location = np.array([-56858.316406,-43683.730469,5117.776855])

constellation_1 = {"cyan": [818.195312, -971.097656, 458.900879], "magenta": [867.039062, -11.128906, 674.893066],
                     "yellow": [867.039062, 698.871094, 524.893066], "red": [767.039062, 308.871094, 164.893066],
                     "green": [17.039062, -631.128906, 634.893066], "blue": [231.492188, 770.839844, 155.332031],
                     "red_magenta": [-664.863281, -416.613281, 423.383301], "orange": [-1042.960938, -1351.128906, 964.893066],
                     "blue_green": [-1052.960938, 78.871094, 584.893066]}

constellation_2 = {"cyan_blue": [-305,305,0], "purple": [305,305,0],
                    "light_magenta": [305,-305,0],"yellow_green": [-305,-305,0]}

constellation_file = open("data_300m_zoom/constellation_1.pkl","wb")
pickle.dump(constellation_1,constellation_file)
constellation_file.close()

constellation_file = open("data_300m_zoom/constellation_2.pkl","wb")
pickle.dump(constellation_2,constellation_file)
constellation_file.close()

pixels_height = 2160
pixels_width = 3840
fov = np.pi / 2

# f = open("FOV", "r")
# fov = float(f.read())
# f.close()

focal_pixels_init = pixels_width / (2 * np.tan(fov / 2))

estimator_cv = PoseEstimator(constellation_1, pixels_width, pixels_height, focal_pixels_init, solver_type="opencv")


## initialize error plots
# error_plots = ["X Error", "Y Error", "Z Error"]
# plotter = ErrorPlotter(error_plots, 50, 'm', 's')
# plotter.set_title("Errors")
#Change default x and y axes labels if needed using ErrorPlotter.set_xlabel(), etc.
pose_plots = [["X", "Y"], ["Z"]]  # Plotting x vs y, and z (2 plots)
pose_plot = PosePlotter(pose_plots, 'm', 's')
pose_plot.set_xlabel(0, "Distance (m)")  # Change default axes labels if needed
pose_plot.set_ylabel(0, "Horizontal Error (m)")
pose_plot.set_ylabel(1, "Vertical Error (m)")

#start the picture grabing and estimations
light_detector = LightDetector()
start = time.time()
first = True
i = 0
flag = False
once = False
f=open("switch","w")
f.write(str(0))
f.close()

while(True):
# while (time.time()-start < 10):
    start_time = time.time()

    if not flag:
        responses = client.simGetImages([
                        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
        fov = np.deg2rad(client.simGetCameraInfo("0").fov)
    else:
        responses = client.simGetImages([
                        airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
        fov = np.deg2rad(client.simGetCameraInfo("3").fov)
    
    # zoom = pixels_width / (2 * np.tan(fov / 2) * focal_pixels_init)
    # print("zoom: " + str(zoom))
    
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

    image_1d = np.frombuffer(image_bytes, dtype=np.uint8) 

    # reshape array to 3 channel image array H X W X 3
    image_rgb = image_1d.reshape(responses[0].height, responses[0].width, 3)

    focal_pixels = pixels_width / (2 * np.tan(fov / 2))
    estimator_cv.setFocalLengthPixels(focal_pixels)

    if not flag:
        pixel_locations = light_detector.detect(image_rgb,"front")
    
    
    print(len(pixel_locations.values()))
    if len(pixel_locations.values()) < 6:
        if not once:
            f=open("switch","w")
            f.write(str(1))
            f.close()
            once = True
            flag = True
            continue
        pixel_locations = light_detector.final_detect(image_rgb,"front")

        pixel_points = []
        constellation_points = []
        pixel_ids = ["cyan_blue","purple","light_magenta","yellow_green"]
        for id in pixel_ids:
            if id in constellation_2:
                pixel_points.append(pixel_locations[id])
                constellation_points.append(constellation_2[id])

        pixel_points = np.array(pixel_points).astype(float)
        constellation_points = np.array(constellation_points).astype(float)

        constellation_points = np.expand_dims(constellation_points, axis=2)
        pixel_points = np.expand_dims(pixel_points, axis=2)
        focal_pixels = pixels_width / (2 * np.tan(fov / 2))
        cam_mat = np.array([[focal_pixels, 0, pixels_width / 2],
                        [0, focal_pixels, pixels_height / 2],
                        [0, 0, 1]])
        distortion = None

        ret, rvecs, tvecs = cv2.solvePnP(constellation_points, pixel_points, cam_mat, distortion,flags=cv2.SOLVEPNP_IPPE_SQUARE)
        last_rotation = cv2.Rodrigues(rvecs)[0]
        last_translation = tvecs

        estimated_camera_location_cv = -last_rotation.T @ last_translation

        # print(position)
        # print(location_estimate)
        # print( "Not enought points detected")
    else:
        estimated_camera_location_cv = estimator_cv.updatePose(pixel_locations)
    
    
    # # write images and other data to file
    # cv2.imwrite(os.path.normpath("data_300m_zoom/pictures/pic_" + str(i) + '.png'), image_rgb)
    
    # actual_positions = np.array([position.x_val, position.y_val,position.z_val])
    # with open('data_300m_zoom/actual_positions.txt','ab') as f:
    #     np.savetxt(f, actual_positions.reshape(1,3), delimiter=',')  
    
    # with open('data_300m_zoom/FOVs.txt','ab') as f:
    #     np.savetxt(f, [fov], delimiter=',')  

    # pixel_file = open("data_300m_zoom/pixel_locations/px_loc_" + str(i) + ".pkl","wb")
    # pickle.dump(pixel_locations,pixel_file)
    # pixel_file.close()
    i += 1 

    x_est_cv = estimated_camera_location_cv[0, 0]
    y_est_cv = estimated_camera_location_cv[1, 0]
    z_est_cv = estimated_camera_location_cv[2, 0]

    x_val = position.x_val 
    y_val = position.y_val
    z_val = position.z_val

    x_error_cv = x_est_cv - x_val
    y_error_cv = y_est_cv - y_val
    z_error_cv = z_est_cv - z_val

    # t = (responses[0].time_stamp - start_time_stamp) / 1000000000
    # x_perc_error_cv = abs(x_error_cv / x_val) * 100
    # # y_perc_error_cv = abs((y_error_cv +1 )/ (1+y_val)) * 100
    # y_perc_error_cv = abs(np.log(np.exp(y_error_cv)))
    # z_perc_error_cv = abs(z_error_cv / z_val) * 100

    # print([x_val,y_val,z_val])
    # print(estimated_camera_location_cv)

    # plotter.update_plot(t, x_error_cv/100, x_perc_error_cv, y_error_cv/100, y_perc_error_cv, z_error_cv/100, z_perc_error_cv)

    d = np.linalg.norm([x_val,y_val,z_val])
    horizontal_error = np.linalg.norm([x_error_cv,y_error_cv])
    pose_plot.update_plot(d/100, 0,horizontal_error/100, 0, z_error_cv/100)
    print("secs: " + str(time.time()-start_time))