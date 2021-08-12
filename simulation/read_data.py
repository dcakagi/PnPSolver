import numpy as np
import cv2
import pickle

#get the constellation data (3D locations of lights)
constellation_file = open("data_300m/constellation.pkl","rb")
constellation = pickle.load(constellation_file)

# Get real position data
actual_position_data = np.loadtxt("data_300m/actual_positions.txt", delimiter=',')

for i in np.arange(343):
    #grab the pixel locations
    pixel_location_file = open("data_300m/pixel_locations/px_loc_" + str(i) + ".pkl", "rb")
    px_locations = pickle.load(pixel_location_file)
    
    # to get the corresponding image 
    # image = cv2.imread("data_300m/pictures/pic_" + str(i))
