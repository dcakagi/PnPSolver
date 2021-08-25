from typing_extensions import TypeAlias
from matplotlib.pyplot import colorbar
import numpy as np
import cv2
import os

from IPython.core.debugger import set_trace

class LightDetector:
    def __init__(self):
        pass


    def which_color(self, hsv, uppers, lowers, key_colors):
        for i in np.arange(len(key_colors)):
            upper = uppers[i]
            lower = lowers[i]
            color = key_colors[i]
            if np.all(np.less_equal(lower, hsv)) and np.all(np.greater_equal(upper, hsv)):
                the_color = key_colors[i]
                return the_color
        # result = np.isclose(hsv,all_colors,atol=20)
        # for i in np.arange(result.shape[0]):
        #     if np.all(result[i,:]):
        #         return(key_colors[i])
        return 'no color recognized'


    def detect(self,image,name):

        if type(image) == bytes:
            np_image = np.frombuffer(image, dtype=np.uint8)

            cv_image = cv2.imdecode(np_image,cv2.IMREAD_COLOR)
        else:
            cv_image = image

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        light_1_upper = np.array([1.4,255,255], dtype = "double")
        light_1_lower = np.array([0.0,75,200], dtype = "double")
        light_1_upper[0] *= 0.5
        light_1_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_1_lower, light_1_upper)
        all_masks = mask

        cv2.imwrite("mask_1.png", mask)

        light_2_upper = np.array([26.0,255,255], dtype = "double")
        light_2_lower = np.array([10.5,75,200], dtype = "double")
        light_2_upper[0] *= 0.5
        light_2_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_2_lower, light_2_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_2.png", mask)

        light_3_upper = np.array([41.3,255,255], dtype = "double")
        light_3_lower = np.array([27.9,75,200], dtype = "double")
        light_3_upper[0] *= 0.5
        light_3_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_3_lower, light_3_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_3.png", mask)

        light_4_upper = np.array([51.6,255,255], dtype = "double")
        light_4_lower = np.array([42.5,75,200], dtype = "double")
        light_4_upper[0] *= 0.5
        light_4_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_4_lower, light_4_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_4.png", mask)

        light_5_upper = np.array([58.8,255,255], dtype = "double")
        light_5_lower = np.array([56.7,75,200], dtype = "double")
        light_5_upper[0] *= 0.5
        light_5_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_5_lower, light_5_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_5.png", mask)

        light_6_upper = np.array([61.1,255,255], dtype = "double")
        light_6_lower = np.array([60.1,75,200], dtype = "double")
        light_6_upper[0] *= 0.5
        light_6_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_6_lower, light_6_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_6.png", mask)

        light_7_upper = np.array([63.4,255,255], dtype = "double")
        light_7_lower = np.array([62.6,75,200], dtype = "double")
        light_7_upper[0] *= 0.5
        light_7_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_7_lower, light_7_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_7.png", mask)

        light_8_upper = np.array([68.3,255,255], dtype = "double")
        light_8_lower = np.array([65.2,75,200], dtype = "double")
        light_8_upper[0] *= 0.5
        light_8_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_8_lower, light_8_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_8.png", mask)

        light_9_upper = np.array([75.6,255,255], dtype = "double")
        light_9_lower = np.array([68.6,75,200], dtype = "double")
        light_9_upper[0] *= 0.5
        light_9_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_9_lower, light_9_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_9.png", mask)

        light_10_upper = np.array([88.7,255,255], dtype = "double")
        light_10_lower = np.array([76.0,75,200], dtype = "double")
        light_10_upper[0] *= 0.5
        light_10_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_10_lower, light_10_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_10.png", mask)

        light_11_upper = np.array([123.6,255,255], dtype = "double")
        light_11_lower = np.array([104.3,75,200], dtype = "double")
        light_11_upper[0] *= 0.5
        light_11_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_11_lower, light_11_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_11.png", mask)

        light_12_upper = np.array([140.0,255,255], dtype = "double")
        light_12_lower = np.array([134.1,75,200], dtype = "double")
        light_12_upper[0] *= 0.5
        light_12_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_12_lower, light_12_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_12.png", mask)

        light_13_upper = np.array([151.0,255,255], dtype = "double")
        light_13_lower = np.array([147.4,75,200], dtype = "double")
        light_13_upper[0] *= 0.5
        light_13_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_13_lower, light_13_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_13.png", mask)

        light_14_upper = np.array([161.6,255,255], dtype = "double")
        light_14_lower = np.array([159.6,75,200], dtype = "double")
        light_14_upper[0] *= 0.5
        light_14_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_14_lower, light_14_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_14.png", mask)

        light_15_upper = np.array([169.0,255,255], dtype = "double")
        light_15_lower = np.array([165.0,75,200], dtype = "double")
        light_15_upper[0] *= 0.5
        light_15_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_15_lower, light_15_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_15.png", mask)

        light_16_upper = np.array([172.9,255,255], dtype = "double")
        light_16_lower = np.array([171.9,75,200], dtype = "double")
        light_16_upper[0] *= 0.5
        light_16_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_16_lower, light_16_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_16.png", mask)

        light_17_upper = np.array([177.0,255,255], dtype = "double")
        light_17_lower = np.array([176.2,75,200], dtype = "double")
        light_17_upper[0] *= 0.5
        light_17_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_17_lower, light_17_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_17.png", mask)

        light_18_upper = np.array([180.9,255,255], dtype = "double")
        light_18_lower = np.array([179.8,75,200], dtype = "double")
        light_18_upper[0] *= 0.5
        light_18_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_18_lower, light_18_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_18.png", mask)

        light_19_upper = np.array([185.3,255,255], dtype = "double")
        light_19_lower = np.array([182.5,75,200], dtype = "double")
        light_19_upper[0] *= 0.5
        light_19_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_19_lower, light_19_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_19.png", mask)

        light_20_upper = np.array([193.4,255,255], dtype = "double")
        light_20_lower = np.array([186.2,75,200], dtype = "double")
        light_20_upper[0] *= 0.5
        light_20_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_20_lower, light_20_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_20.png", mask)

        light_21_upper = np.array([208.4,255,255], dtype = "double")
        light_21_lower = np.array([194.8,75,200], dtype = "double")
        light_21_upper[0] *= 0.5
        light_21_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_21_lower, light_21_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_21.png", mask)

        light_22_upper = np.array([241.5,255,255], dtype = "double")
        light_22_lower = np.array([224.5,75,200], dtype = "double")
        light_22_upper[0] *= 0.5
        light_22_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_22_lower, light_22_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_22.png", mask)

        light_23_upper = np.array([259.2,255,255], dtype = "double")
        light_23_lower = np.array([258.2,75,200], dtype = "double")
        light_23_upper[0] *= 0.5
        light_23_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_23_lower, light_23_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_23.png", mask)

        light_24_upper = np.array([275.3,255,255], dtype = "double")
        light_24_lower = np.array([270.4,75,200], dtype = "double")
        light_24_upper[0] *= 0.5
        light_24_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_24_lower, light_24_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_24.png", mask)

        light_25_upper = np.array([294.0,255,255], dtype = "double")
        light_25_lower = np.array([285.5,75,200], dtype = "double")
        light_25_upper[0] *= 0.5
        light_25_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_25_lower, light_25_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_25.png", mask)

        light_26_upper = np.array([302.5,255,255], dtype = "double")
        light_26_lower = np.array([295.5,75,200], dtype = "double")
        light_26_upper[0] *= 0.5
        light_26_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_26_lower, light_26_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_26.png", mask)

        light_27_upper = np.array([309.2,255,255], dtype = "double")
        light_27_lower = np.array([308.6,75,200], dtype = "double")
        light_27_upper[0] *= 0.5
        light_27_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_27_lower, light_27_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_27.png", mask)

        light_28_upper = np.array([313.6,255,255], dtype = "double")
        light_28_lower = np.array([310.3,75,200], dtype = "double")
        light_28_upper[0] *= 0.5
        light_28_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_28_lower, light_28_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_28.png", mask)

        light_29_upper = np.array([335.9,255,255], dtype = "double")
        light_29_lower = np.array([323.1,75,200], dtype = "double")
        light_29_upper[0] *= 0.5
        light_29_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_29_lower, light_29_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_29.png", mask)

        light_30_upper = np.array([345.2,255,255], dtype = "double")
        light_30_lower = np.array([337.2,75,200], dtype = "double")
        light_30_upper[0] *= 0.5
        light_30_lower[0] *= 0.5
        mask = cv2.inRange(hsv_image, light_30_lower, light_30_upper)
        all_masks = cv2.add(all_masks,mask)

        cv2.imwrite("mask_30.png", mask)

        uppers = np.array([light_1_upper, light_2_upper, light_3_upper, light_4_upper, light_5_upper, \
            light_6_upper, light_7_upper, light_8_upper, light_9_upper, light_10_upper, light_11_upper, \
            light_12_upper, light_13_upper, light_14_upper, light_15_upper, light_16_upper, light_17_upper, \
            light_18_upper, light_19_upper, light_20_upper, light_21_upper, light_22_upper, light_23_upper, \
            light_24_upper, light_25_upper, light_26_upper, light_27_upper, light_28_upper, light_29_upper, \
            light_30_upper])
        lowers = np.array([light_1_lower, light_2_lower, light_3_lower, light_4_lower, light_5_lower, \
            light_6_lower, light_7_lower, light_8_lower, light_9_lower, light_10_lower, light_11_lower, \
            light_12_lower, light_13_lower, light_14_lower, light_15_lower, light_16_lower, light_17_lower, \
            light_18_lower, light_19_lower, light_20_lower, light_21_lower, light_22_lower, light_23_lower, \
            light_24_lower, light_25_lower, light_26_lower, light_27_lower, light_28_lower, light_29_lower, \
            light_30_lower])
        key_colors = np.array(["light_1", "light_2", "light_3", "light_4", "light_5", "light_6", "light_7", \
            "light_8", "light_9", "light_10", "light_11", "light_12", "light_13", "light_14", "light_15", "light_16", \
            "light_17", "light_18", "light_19", "light_20", "light_21", "light_22", "light_23", "light_24", "light_25", \
            "light_26", "light_27", "light_28", "light_29", "light_30"])

        # output = cv2.bitwise_and(cv_image, cv_image, mask = all_masks)


        # # kernel=np.ones((3,3),np.uint8)
        # # dilated=cv2.dilate(output,kernel,iterations=3)
        # gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # apply binary thresholding
        img2 = all_masks.copy()
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(all_masks, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        for i in range(0, nb_components):
            if sizes[i] <= 10:
                img2[output == i + 1] = 0
        kernel = np.ones((5,5), np.uint8)
        img2 = cv2.dilate(img2, kernel, iterations=1)

        # img2 = cv2.blur(img2, (3, 3),0)
        # ret, thresh = cv2.threshold(img2, 25, 255, cv2.THRESH_BINARY)
        # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        # visualize the binary image
        # cv2.imshow('Binary image', img2)
        # cv2.waitKey(1)

        # img2 = cv2.bitwise_not(img2)
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        params.blobColor = 255

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 256

        # # Filter by Area.
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 10000000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5

        # Filter by Inertia
        params.filterByInertia =True
        params.minInertiaRatio = 0.5

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(img2)

        # small = cv2.resize(im_with_keypoints, (0,0), fx=0.5, fy=0.5)
        # cv2.imshow(name,small)
        # cv2.waitKey(1)

        image_copy = cv_image.copy()
        pixel_locations = {}
        for point in keypoints:

                x = int(point.pt[0])
                y = int(point.pt[1])
                l = int(point.size/1.9)

                cv2.rectangle(image_copy, (x - l, y - l), (x + l, y + l), (0, 0, 255), 2)

                cx = point.pt[0]
                cy = point.pt[1]

                cx_int = int(cx)
                cy_int = int(cy)
                # msg = "({}, {})".format(cx_int,cy_int)
                # print(msg)
                # print("color: HSV " + str(hsv_image[cy_int,cx_int,:]) + "--" + self.which_color(hsv_image[cy_int,cx_int,:],all_colors,uppers,lowers,key_colors))
                # print()
                color_name = self.which_color(hsv_image[cy_int,cx_int,:],uppers,lowers,key_colors)
                pixel_locations[color_name] = [cx,cy]

        # # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE

        # contours, hierarchy = cv2.findContours(image=img2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        # # draw contours on the original image
        # image_copy = cv_image.copy()
        # pixel_locations = {}
        # # print("pixel locations of lights from top left:")
        # print(len(contours))
        # for cnt in contours:

        #         rect = cv2.minAreaRect(cnt)
        #         box = cv2.boxPoints(rect)
        #         box = np.int0(box)
        #         cv2.drawContours(image_copy,[box],0,(0,0,255),3)

        #         M = cv2.moments(cnt)
        #         if M['m00'] == 0:
        #             continue
        #         cx = float(M['m10']/M['m00'])
        #         cy = float(M['m01']/M['m00'])

        #         cx_int = int(cx)
        #         cy_int = int(cy)
        #         # msg = "({}, {})".format(cx_int,cy_int)
        #         # print(msg)
        #         # print("color: HSV " + str(hsv_image[cy_int,cx_int,:]) + "--" + self.which_color(hsv_image[cy_int,cx_int,:],all_colors,uppers,lowers,key_colors))
        #         # print()
        #         color_name = self.which_color(hsv_image[cy_int,cx_int,:],all_colors,uppers,lowers,key_colors)
        #         pixel_locations[color_name] = [cx,cy]

        # cv2.imwrite(os.path.normpath("points_detected" + '.png'), cv_image)

        # small = cv2.resize(image_copy, (0,0), fx=0.5, fy=0.5)
        # cv2.namedWindow(name)        # Create a named window
        # cv2.moveWindow(name, 0,0)  # Move it to (40,30)
        # cv2.imshow(name, small)
        # cv2.waitKey(1)
        # print("got it")
        # return None
        return pixel_locations