import airsim
from airsim.types import DistanceSensorData
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

import matplotlib.pyplot as plt
plt.ion()


def saturate(fov):
    if fov > 120.0:
        fov = 120

    return fov
def process_im(camera):
    responses = client.simGetImages([
                        airsim.ImageRequest(camera, airsim.ImageType.Scene,False,False)])  #scene vision image in uncompressed RGBA array
        # fov = np.deg2rad(client.simGetCameraInfo("0").fov)
   
    # zoom = pixels_width / (2 * np.tan(fov / 2) * focal_pixels_init)
    # print("zoom: " + str(zoom))

   
    position = responses[0].camera_position
    orientation = responses[0].camera_orientation
    image_bytes = responses[0].image_data_uint8

    position.x_val *= 100
    position.y_val *= 100
    position.z_val *= -100

    position.z_val -= 140 # start location is 140cm off ground

    # position.x_val += start_location[0]
    # position.y_val += start_location[1]
    # position.z_val += start_location[2]

    # position.x_val -= landing_point[0]
    # position.y_val -= landing_point[1]
    # position.z_val -= landing_point[2]

    # print("camera Position: ")
    # print(position)
    # print()

    image_1d = np.frombuffer(image_bytes, dtype=np.uint8)

    # reshape array to 3 channel image array H X W X 3
    image_rgb = image_1d.reshape(responses[0].height, responses[0].width, 3)

    # focal_pixels = pixels_width / (2 * np.tan(fov / 2))
    # estimator_cv.setFocalLengthPixels(focal_pixels)

    pixel_locations = light_detector.detect(image_rgb,"front")
    return pixel_locations, position

if __name__ == '__main__':
    # connect to the AirSim simulator
    # client = airsim.TiltrotorClient("10.32.114.184")
    client = airsim.TiltrotorClient("10.32.114.193")
    client.confirmConnection()

    # landing_point = np.array([-35880.0,-41940.0,4870.0])
    # start_location = np.array([-56858.316406,-43683.730469,5117.776855])

    landing_point = np.array([-24740,	-44980,	1980])
    start_location = landing_point.copy()

    # constellation_1 = {"cyan": [818.195312, -971.097656, 458.900879], "magenta": [867.039062, -11.128906, 674.893066],
    #                      "yellow": [867.039062, 698.871094, 524.893066], "red": [767.039062, 308.871094, 164.893066],
    #                      "green": [17.039062, -631.128906, 634.893066], "blue": [231.492188, 770.839844, 155.332031],
    #                      "red_magenta": [-664.863281, -416.613281, 423.383301], "orange": [-1042.960938, -1351.128906, 964.893066],
    #                      "blue_green": [-1052.960938, 78.871094, 584.893066]}

    constellation_1 = {"1" :  [3020, 0, 760],
            "2": [3000, -1100,720],
            "3": [2800, -2180,620],
            "4": [2160, -2440,  370],
            "5": [1530, -2600, 130],
            "6": [430,  -2370, 70],
            "7":[ -540,   -1890, 80],
            "8":[ -1350,  -1120, 90],
            "9":[ -1800,  0,     70],
            "10":[ -1350,  1120,  90],
            "11":[ -540,   1890,  80],
            "12":[ 430,    2370,  70],
            "13":[ 1530,   2600,  130],
            "14":[ 2160,   2440,  370],
            "15":[ 2800,   2180,  620],
            "16":[ 3000,   1100,  720]}

    constellation_2 = {"18": [310,   0,    -200],
            "19": [95,    0,    -200],
            "20": [-115,  0,    -200],
            "21": [-330,  0,    -200],
            "22": [95,    -100, -200],
            "23": [-115,  -100, -200],
            "24": [-240,  -100, -200],
            "25": [95,    100,  -200],
            "26": [-115,  100,  -200],
            "27": [-240,  100,  -200],
            "28": [240,   120,  -200],
            "29": [130,   200,  -200],
            "30": [-30,   260,  -200],
            "31": [-210,  270,  -200],
            "32": [240,   -120, -200],
            "33": [130,   -200, -200],
            "34": [-30,   -260, -200],
            "35": [-210,  -270, -200],


            "36": [660,   0,    -200],
            "37": [440,   0,    -200],
            "38": [-460,  0,    -200],
            "39": [-630,  0,    -200],
            "40": [-860,  0,    -200],
            "41": [440,   -250, -200],
            "42": [-460,  -250, -200],
            "43": [-630,  -250, -200],
            "44": [440,   250,  -200],
            "45": [-460,  250,  -200],
            "46": [-630,  250,  -200],
            "47": [850,   120,  -200],
            "48": [850,   -120, -200],
            "49": [660,   300,  -200],
            "50": [660,   -300, -200],
            "51": [440,   500,  -200],
            "52": [440,   -500, -200],
            "53": [100,   620,  -200],
            "54": [100,   -620, -200],
            "55": [-230,  680,  -200],
            "56": [-230,  -680, -200],
            "57": [-540,  690, -200],
            "58": [-540,  -690, -200]}

    # constellation_2 = {"cyan_blue": [-305,305,0], "purple": [305,305,0],
    #                     "light_magenta": [305,-305,0],"yellow_green": [-305,-305,0]}

    # constellation_file = open("data_300m_zoom/constellation_1.pkl","wb")
    # pickle.dump(constellation_1,constellation_file)
    # constellation_file.close()

    # constellation_file = open("data_300m_zoom/constellation_2.pkl","wb")
    # pickle.dump(constellation_2,constellation_file)
    # constellation_file.close()

    pixels_height = 2160
    pixels_width = 3840
    fov = np.pi / 2

    # f = open("FOV", "r")
    # fov = float(f.read())
    # f.close()

    focal_pixels_init = pixels_width / (2 * np.tan(fov / 2))

    estimator_cv = PoseEstimator(constellation_1, pixels_width, pixels_height, focal_pixels_init, solver_type="opencv",use_guess=True)


    # f = open("FOV", "r")
    # fov = float(f.r

    ## initialize error plots
    # error_plots = ["X Error", "Y Error", "Z Error"]
    # plotter = ErrorPlotter(error_plots, 50, 'm', 's')
    # plotter.set_title("Errors")
    #Change default x and y axes labels if needed using ErrorPlotter.set_xlabel(), etc.
    # pose_plots = [["X", "Y"], ["Z"]]  # Plotting x vs y, and z (2 plots)
    # pose_plot = PosePlotter(pose_plots, 'm', 's')
    # pose_plot.set_xlabel(0, "Distance (m)")  # Change default axes labels if needed
    # pose_plot.set_ylabel(0, "Horizontal Error (m)")
    # pose_plot.set_ylabel(1, "Vertical Error (m)")

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

    end_wrt_start = np.array([0, 0, 0])

    # zoom = 1.9
    # width = 3840
    # FOV = 90
    # focal_length = width / (2 * np.tan(np.deg2rad(FOV)/2))
    # client.simSetCameraFov("3",120)
    client.simSetCameraFov("3",120)
    client.simSetCameraFov("0",90)
    # f=open("switch","w")
    # f.write(str(0))
    # f.close()

    height = 15*100
    dist = 450*100
    # start = np.subtract(end, [dist, 0, np.tan(np.deg2rad(7))*dist-height])

    startx = end_wrt_start[0] - dist
    starty = end_wrt_start[1]
    startz = end_wrt_start[2] - height - np.arctan(np.deg2rad(7))*dist

    dx = 100
    xs = np.arange(startx,end_wrt_start[0]+dx,dx)
    ys = np.array([end_wrt_start[1]]*len(xs))

    x = np.arange(-(len(xs)-1),dx,dx)
    zs = np.arctan(np.deg2rad(7))*xs - height + end_wrt_start[2]

    waypoints = np.array([xs,ys,zs],dtype=float)
    waypoints = np.swapaxes(waypoints,0,1)

    N = 100
    zs = np.arange(zs[-1],end_wrt_start[2],10)
    xs = np.array([xs[-1]]*len(zs))
    ys = np.array([ys[-1]]*len(zs))

    end_waypoints = np.array([xs,ys,zs],dtype=float)
    end_waypoints = np.swapaxes(end_waypoints,0,1)
    end_waypoints = np.vstack((end_waypoints,np.array(end_waypoints[-1].tolist()*5000).reshape(5000,3)))
    # waypoints = np.vstack((waypoints,end_waypoints))
    # waypoints = end_waypoints
    # waypoints = [waypoints[-1]]
    destination_point = landing_point - start_location

    # while(True):
    # while (time.time()-start < 10):
    cam = "0"
    first = True

    # plt.xlim([-50, 400])
    # plt.ylim([-1,1])
    vertical_errors = []
    horizontal_errors = []
    distances = []
    switch = False
    only_once = True
    for i in range(len(waypoints)):
    # for w in waypoints:
        w = waypoints[i]
        start_time = time.time()
        position = airsim.Vector3r(w[0]/100,w[1]/100,w[2]/100)
        if not flag:
            rot = airsim.to_quaternion(np.deg2rad(-7.0),0,0)
        else:
            rot = airsim.to_quaternion(0,0,0)

        orientation = rot
        pose = airsim.Pose(position,orientation)
        start = time.time()
        total = .01
        tilt =  [float("NaN"),float("NaN"),float("NaN")]

        # while(True):
        # while(time.time()-start < total):
        client.simSetTiltrotorPose(pose,tilt,True,vehicle_name='0',spin_props=True)
        
        px_locations, position = process_im(cam)

        # if abs(position.x_val) < 300 and only_once:
        #     only_once = False
        #     cam = "3"
        #     px_locations, position = process_im(cam)
        #     initial_estimator = PoseEstimator(constellation_2, pixels_width, pixels_height, focal_pixels_init, solver_type="opencv", method=cv2.SOLVEPNP_EPNP)
        #     init_guess = initial_estimator.updatePose(px_locations)
        #     estimator_cv.last_rotation = -initial_estimator.last_rotation
        #     estimator_cv.last_translation = -initial_estimator.last_translation
        #     estimator_cv.constellation = constellation_2
        #     continue
        print(len(px_locations.values()))
        # if len(px_locations.values()) < 16 and first:
        #     first = False
        #     client.simSetCameraFov(cam,90)
        #     px_locations,position = process_im(cam)
        #     estimator_cv.setFocalLengthPixelsFromFOV(np.deg2rad(90))
        #     estimated_camera_location_cv = estimator_cv.updatePose(px_locations)
        #     switch = True
        # else:
        estimated_camera_location_cv = estimator_cv.updatePose(px_locations)


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

        vertical_errors.append(abs(z_error_cv/100))
        horizontal_errors.append(abs(np.linalg.norm([x_error_cv/100,y_error_cv/100])))
        distances.append(np.linalg.norm([x_val/100,y_val/100]))
        if switch:
            switch = False
            the_switch_x = distances[-1]
            the_switch_y1 = vertical_errors[-1]
            the_switch_y2 = horizontal_errors[-1]

        # t = (responses[0].time_stamp - start_time_stamp) / 1000000000
        # x_perc_error_cv = abs(x_error_cv / x_val) * 100
        # # y_perc_error_cv = abs((y_error_cv +1 )/ (1+y_val)) * 100
        # y_perc_error_cv = abs(np.log(np.exp(y_error_cv)))
        # z_perc_error_cv = abs(z_error_cv / z_val) * 100

        # print([x_val,y_val,z_val])
        # print(estimated_camera_location_cv)

        # plotter.update_plot(t, x_error_cv/100, x_perc_error_cv, y_error_cv/100, y_perc_error_cv, z_error_cv/100, z_perc_error_cv)

        # d = np.linalg.norm([x_val,y_val,z_val])
        # horizontal_error = np.linalg.norm([x_error_cv,y_error_cv])
        # pose_plot.update_plot(d/100, 0,horizontal_error/100, 0, z_error_cv/100)
        print("x_Error: " + str(x_error_cv/100))
        print("Y_Error: " + str(y_error_cv/100))
        print("Z_Error: " + str(z_error_cv/100))

        print("secs: " + str(time.time()-start_time))
    fig, axs = plt.subplots(1,2,figsize=(1000/96, 500/96))
    axs[1].plot(distances, vertical_errors)
    axs[1].plot(distances, vertical_errors)
    axs[1].yaxis.grid()
    axs[1].set_xlabel("range (m)")
    axs[1].set_ylabel("vertical error (m)")
    # axs[1].set_ylim([0,max([max(vertical_errors),max(horizontal_errors)])+.2])
    axs[1].set_ylim([-0.05,1.2])
    # axs[1].annotate('Switch to\n110' + u"\N{DEGREE SIGN}" + ' FOV Cam', xy=(the_switch_x,the_switch_y1),  xycoords='data',
    #         xytext=(the_switch_x+50,the_switch_y1-0.3), textcoords='data',
    #         arrowprops=dict(facecolor='black', shrink=0.01,width=2),
    #         horizontalalignment='center', verticalalignment='top',
    #         )
    
    axs[0].plot(distances, horizontal_errors)
    axs[0].yaxis.grid()
    axs[0].set_xlabel("range (m)")
    axs[0].set_ylabel("horizontal error (m)")
    # axs[0].set_ylim([0,max([max(vertical_errors),max(horizontal_errors)])+.2])
    axs[0].set_ylim([-0.05,1.2])
    # axs[0].annotate('Switch to\n110' + u"\N{DEGREE SIGN}" + ' FOV Cam', xy=(the_switch_x,the_switch_y2),  xycoords='data',
    #         xytext=(the_switch_x+50,the_switch_y2+.3), textcoords='data',
    #         arrowprops=dict(facecolor='black', shrink=0.01,width=2),
    #         horizontalalignment='center', verticalalignment='top',
    #         )
    plt.savefig("450m_Approach_Simulated.png", dpi=500)
    plt.show()  