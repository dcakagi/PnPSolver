from typing import Counter
from msgpackrpc.transport.tcp import ClientTransport
# import setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2
from numpy.lib.type_check import imag
import time

import sys
sys.path.append("../")

from PoseEstimator import PoseEstimator
from Plotter import ErrorPlotter
from Plotter import PosePlotter

class LightDetector:
    def __init__(self):
        pass
    
    def upper (self, rgb):
        # if (rgb.count(255) == 2 or rgb.count(0) == 2):
        #     adjust_brightness = True
        # else: 
        #     adjust_brightness = False
        
        new_rgb = rgb.copy()
        for i in range(3):
            # if new_rgb[i] == 0 and adjust_brightness:
            #     new_rgb[i] += 150
            # else:
            new_rgb[i] += 25

            if new_rgb[i] >= 255:
                new_rgb[i] = 255

        new_rgb = np.array(new_rgb, dtype = "uint8")
        return new_rgb

    def lower(self, rgb):
        new_rgb = rgb.copy()
        for i in range(3):
            new_rgb[i] -= 25
            if new_rgb[i] <= 0:
                new_rgb[i] = 0
            
        new_rgb = np.array(new_rgb, dtype = "uint8")
        return new_rgb

    def which_color(self, rgb,all_colors,key_colors):
        result = np.isclose(rgb,all_colors,atol=30)
        for i in np.arange(result.shape[0]):
            if np.all(result[i,:]):
                return(key_colors[i])

        return 'no color recognized'


    def detect(self,image,name):

        if type(image) == bytes:
            np_image = np.frombuffer(image, dtype=np.uint8)

            cv_image = cv2.imdecode(np_image,cv2.IMREAD_COLOR)
        else:
            cv_image = image

        cyan = [252,255,25]
        lower_cyan = self.lower(cyan)
        upper_cyan = self.upper(cyan)
        mask = cv2.inRange(cv_image, lower_cyan, upper_cyan)
        all_masks = mask

        magenta = [234,37,245]
        lower_magenta = self.lower(magenta)
        upper_magenta= self.upper(magenta)
        mask = cv2.inRange(cv_image, lower_magenta, upper_magenta)
        all_masks = cv2.add(all_masks,mask)

        yellow = [45,253,250]
        lower_yellow = self.lower(yellow)
        upper_yellow = self.upper(yellow)
        mask = cv2.inRange(cv_image, lower_yellow, upper_yellow)
        all_masks = cv2.add(all_masks,mask)
        
        red = [0,0,255]
        lower_red = self.lower(red)
        upper_red = self.upper(red)
        mask = cv2.inRange(cv_image, lower_red, upper_red)
        all_masks = cv2.add(all_masks,mask)
        
        green = [25,255,0]
        lower_green = self.lower(green)
        upper_green = self.upper(green)
        mask = cv2.inRange(cv_image, lower_green, upper_green)
        all_masks = cv2.add(all_masks,mask)
        
        blue = [238,71,3]
        lower_blue = self.lower(blue)
        upper_blue = self.upper(blue)
        mask = cv2.inRange(cv_image, lower_blue, upper_blue)
        all_masks = cv2.add(all_masks,mask)
        
        red_magenta = [133,0,255]
        lower_redmagenta = self.lower(red_magenta)
        upper_redmagenta = self.upper(red_magenta)
        mask = cv2.inRange(cv_image, lower_redmagenta, upper_redmagenta)
        all_masks = cv2.add(all_masks,mask)
        
        orange = [0,130,255]
        lower_orange = self.lower(orange)
        upper_orange = self.upper(orange)
        mask = cv2.inRange(cv_image, lower_orange, upper_orange)
        all_masks = cv2.add(all_masks,mask)

        blue_green = [155,255,0]
        lower_bluegreen = self.lower(blue_green)
        upper_bluegreen = self.upper(blue_green)
        mask = cv2.inRange(cv_image, lower_bluegreen, upper_bluegreen)
        all_masks = cv2.add(all_masks,mask)

        all_colors = np.array([cyan,magenta,yellow,red,green,blue,red_magenta,orange,blue_green])
        key_colors = np.array(["cyan","magenta","yellow","red","green","blue","red_magenta","orange","blue_green"])
    
        output = cv2.bitwise_and(cv_image, cv_image, mask = all_masks)
        
        
        # kernel=np.ones((3,3),np.uint8)
        # dilated=cv2.dilate(output,kernel,iterations=3)
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # apply binary thresholding
        img2 = gray.copy()
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        for i in range(0, nb_components):
            if sizes[i] <= 3:
                img2[output == i + 1] = 0
        img2 = cv2.blur(img2, (3, 3))
        ret, thresh = cv2.threshold(img2, 25, 255, cv2.THRESH_BINARY)
        # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        # # visualize the binary image
        # cv2.imshow('Binary image', thresh)
        # cv2.waitKey(1)

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                            
        # draw contours on the original image
        image_copy = cv_image.copy()
        pixel_locations = {}
        # print("pixel locations of lights from top left:")
        for cnt in contours:

                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image_copy,[box],0,(0,0,255),3)

                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                msg = "({}, {})".format(cx,cy)
                print(msg)
                print("color: BGR " + str(cv_image[cy,cx,:]) + "--" + self.which_color(cv_image[cy,cx,:],all_colors,key_colors))
                print()
                color_name = self.which_color(cv_image[cy,cx,:],all_colors,key_colors)
                pixel_locations[color_name] = [cx,cy]

        # print("==============================================================================")

        cv2.imwrite(os.path.normpath("points_detected" + '.png'), cv_image)
        # gpuMat = cv2.cuda_GpuMat()
        # gpuMat.upload(image_copy)

        # cv2.imshow(name, gpuMat.download())
        # cv2.waitKey(1)
        small = cv2.resize(image_copy, (0,0), fx=0.5, fy=0.5) 
        # cv2.imshow(name, small)
        # cv2.waitKey(1)
        cv2.namedWindow(name)        # Create a named window
        cv2.moveWindow(name, 0,0)  # Move it to (40,30)
        cv2.imshow(name, small)
        cv2.waitKey(1)        
        # print("got it")
        return pixel_locations




# connect to the AirSim simulator
client = airsim.MultirotorClient("10.32.114.184")
client.confirmConnection()

landing_point = np.array([-35880.0,-41940.0,4870.0])
start_location = np.array([-56858.316406,-43683.730469,5117.776855])

constellation = {"cyan": [818.195312, -971.097656, 458.900879], "magenta": [867.039062, -11.128906, 674.893066],
                     "yellow": [867.039062, 698.871094, 524.893066], "red": [767.039062, 308.871094, 164.893066],
                     "green": [17.039062, -631.128906, 634.893066], "blue": [231.492188, 770.839844, 155.332031],
                     "red_magenta": [-664.863281, -416.613281, 423.383301], "orange": [-1042.960938, -1351.128906, 964.893066],
                     "blue_green": [-1052.960938, 78.871094, 584.893066]}
                     
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
while(True):
# while (time.time()-start < 10):
    start_time = time.time()

    # client.simPause(True)

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

    # reshape array to 4 channel image array H X W X 4
    image_rgb = image_1d.reshape(responses[0].height, responses[0].width, 3)
    
    # write images to file as picture or video
    cv2.imwrite(os.path.normpath("pics_300m/pic_" + str(i) + '.png'), image_rgb)
    i += 1 
    # video_out.write(image_rgb)

    # # original image is fliped vertically
    # image_rgb = np.flipud(image_rgb)

    # image = client.simGetImage("0",airsim.ImageType.Scene,vehicle_name='0')
    # pixel_locations = light_detector.detect(image_rgb,"front")
   
    # print(len(pixel_locations.values()))
    # if len(pixel_locations.values()) < 6:
    #     print( "Not enought points detected")
    #     continue
    # # estimated_camera_location = estimator.updatePose(pixel_locations)
    # estimated_camera_location_cv = estimator_cv.updatePose(pixel_locations)

    # # x_est_cv = estimated_camera_location[0, 0]
    # # y_est_cv = estimated_camera_location[1, 0]
    # # z_est_cv = estimated_camera_location[2, 0]
    # x_est_cv = estimated_camera_location_cv[0, 0]
    # y_est_cv = estimated_camera_location_cv[1, 0]
    # z_est_cv = estimated_camera_location_cv[2, 0]

    # x_val = position.x_val 
    # y_val = position.y_val
    # z_val = position.z_val

    # x_error_cv = x_est_cv - x_val
    # y_error_cv = y_est_cv - y_val
    # z_error_cv = z_est_cv - z_val
    # # x_error_cv = x_est_cv - x_val
    # # y_error_cv = y_est_cv - y_val
    # # z_error_cv = z_est_cv - z_val

    # # print(f"\nOPnP X estimate: {x_est}")
    # # print(f"OPnP Y estimate: {y_est}")
    # # print(f"OPnP Z estimate: {z_est}")

    # # print(f"\nOPnP X % error: {x_error}")
    # # print(f"OPnP Y % error: {y_error}")
    # # print(f"OPnP Z % error: {z_error}")

    # # print(f"\nOpencv X estimate: {x_est_cv}")
    # # print(f"Opencv Y estimate: {y_est_cv}")
    # # print(f"Opencv Z estimate: {z_est_cv}")

    # # print(f"\nOpencv X % error: {x_error_cv}")
    # # print(f"Opencv Y % error: {y_error_cv}")
    # # print(f"Opencv Z % error: {z_error_cv}")

    # t = (responses[0].time_stamp - start_time_stamp) / 1000000000
    # x_perc_error_cv = abs(x_error_cv / x_val) * 100
    # # y_perc_error_cv = abs((y_error_cv +1 )/ (1+y_val)) * 100
    # y_perc_error_cv = abs(np.log(np.exp(y_error_cv)))
    # z_perc_error_cv = abs(z_error_cv / z_val) * 100

    # # print(f"\nOpencv X % error: {x_perc_error_cv}")
    # # print(f"Opencv Y % error: {y_perc_error_cv}")
    # # print(f"Opencv Z % error: {z_perc_error_cv}")

    # # plotter.update_plot(t, x_error_cv/100, x_perc_error_cv, y_error_cv/100, y_perc_error_cv, z_error_cv/100, z_perc_error_cv)
    # pose_plot.update_plot(t, x_val/100, x_est_cv/100, y_val/100, y_est_cv/100, z_val/100, z_est_cv/100)
    # # print("updated")
    print("secs: " + str(time.time()-start_time))
