from matplotlib.pyplot import colorbar
import numpy as np
import cv2
import os
class LightDetector:
    def __init__(self):
        pass

    def upper (self, rgb,to_add):
        if (rgb.count(255) == 2 or rgb.count(0) == 2):
            adjust_brightness = True
        else:
            adjust_brightness = False

        new_rgb = rgb.copy()
        new_hsv = np.add(new_rgb, to_add)

        if new_hsv[0] >=180:
            new_hsv[0] = 180

        if new_hsv[1] >= 255:
            new_hsv[1] = 255

        if new_hsv[2] >= 255:
            new_hsv[2] = 255


        new_rgb = np.array(new_hsv, dtype = "uint8")
        return new_rgb

    def lower(self, rgb,sub):
        new_rgb = rgb.copy()
        new_hsv = np.subtract(new_rgb, sub)
        if new_hsv[0] <=0:
            new_hsv[0] = 0

        if new_hsv[1] <= 0:
            new_hsv[1] = 0

        if new_hsv[2] <= 0:
            new_hsv[2] = 0

        new_rgb = np.array(new_hsv, dtype = "uint8")
        return new_rgb

    def which_color(self, hsv,all_colors,uppers,lowers,key_colors):
        for i in np.arange(len(all_colors)):
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
        # cv2.imshow("conversion",hsv_image)
        # cv2.waitKey(1)
        # cyan = [252,255,40]
        # cyan = [255,255,0]
        cyan = [90,255, 225]
        lower_cyan = self.lower(cyan,[3,75,30])
        upper_cyan = self.upper(cyan,[3,10,30])
        mask = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
        all_masks = mask

        # magenta = [234,37,245]
        # magenta = [255,0,255]
        magenta = [150,255,230]
        lower_magenta = self.lower(magenta,[3,75,30])
        upper_magenta= self.upper(magenta,[3,10,30])
        mask = cv2.inRange(hsv_image, lower_magenta, upper_magenta)
        all_masks = cv2.add(all_masks,mask)

        # yellow = [45,253,250]
        # yellow = [0,255,255]
        yellow = [30,255,230]
        lower_yellow = self.lower(yellow,[3,75,30])
        upper_yellow = self.upper(yellow,[3,10,30])
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        all_masks = cv2.add(all_masks,mask)

        # red = [0,0,255]
        red = [0,255,255]
        lower_red = self.lower(red,[2,30,20])
        upper_red = self.upper(red,[2,10,20])
        lower_red2 = lower_red.copy()
        lower_red2[0] = 177
        upper_red2 = upper_red.copy()
        upper_red2[0] = 180
        mask = cv2.inRange(hsv_image, lower_red2, upper_red2)
        all_masks = cv2.add(all_masks,mask)
        mask = cv2.inRange(hsv_image, lower_red, upper_red)
        all_masks = cv2.add(all_masks,mask)

        # green = [25,255,0]
        # green = [0,255,0]
        green = [62,255,255]
        lower_green = self.lower(green,[3,10,55])
        upper_green = self.upper(green,[3,10,30])
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        all_masks = cv2.add(all_masks,mask)

        # blue = [238,71,3]
        # blue = [255,0,0]
        blue = [117,255,230]
        lower_blue = self.lower(blue,[7,50,30])
        upper_blue = self.upper(blue,[4,10,30])
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        all_masks = cv2.add(all_masks,mask)

        # red_magenta = [133,0,255]
        red_magenta = [165,255,255]
        lower_redmagenta = self.lower(red_magenta,[2,50,30])
        upper_redmagenta = self.upper(red_magenta,[2,10,30])
        mask = cv2.inRange(hsv_image, lower_redmagenta, upper_redmagenta)
        all_masks = cv2.add(all_masks,mask)

        # orange = [0,130,255]
        orange = [15,255,245]
        lower_orange = self.lower(orange,[1,10,30])
        upper_orange = self.upper(orange,[1,10,30])
        mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
        all_masks = cv2.add(all_masks,mask)

        # blue_green = [155,255,0]
        blue_green = [74,255,255]
        lower_bluegreen = self.lower(blue_green,[3,10,10])
        upper_bluegreen = self.upper(blue_green,[5,10,10])
        mask = cv2.inRange(hsv_image, lower_bluegreen, upper_bluegreen)
        all_masks = cv2.add(all_masks,mask)

        all_colors = np.array([cyan,magenta,yellow,red,green,blue,red_magenta,orange,blue_green])
        uppers = np.array([upper_cyan,upper_magenta,upper_yellow,upper_red,upper_green,upper_blue,upper_redmagenta,upper_orange,upper_bluegreen])
        lowers = np.array([lower_cyan,lower_magenta,lower_yellow,lower_red,lower_green,lower_blue,lower_redmagenta,lower_orange,lower_bluegreen])
        key_colors = np.array(["cyan","magenta","yellow","red","green","blue","red_magenta","orange","blue_green"])

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
                color_name = self.which_color(hsv_image[cy_int,cx_int,:],all_colors,uppers,lowers,key_colors)
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

    def final_detect(self,image,name):

        if type(image) == bytes:
            np_image = np.frombuffer(image, dtype=np.uint8)

            cv_image = cv2.imdecode(np_image,cv2.IMREAD_COLOR)
        else:
            cv_image = image

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # cv2.imshow("conversion",hsv_image)
        # cv2.waitKey(1)
        yellow_green = [39,245, 255]
        lower_yellow_green = self.lower(yellow_green,[3,40,30])
        upper_yellow_green = self.upper(yellow_green,[15,10,30])
        mask = cv2.inRange(hsv_image, lower_yellow_green, upper_yellow_green)
        all_masks = mask

        cyan_blue = [101,245,245]
        lower_cyan_blue = self.lower(cyan_blue,[3,30,40])
        upper_cyan_blue= self.upper(cyan_blue,[11,10,40])
        mask = cv2.inRange(hsv_image, lower_cyan_blue, upper_cyan_blue)
        all_masks = cv2.add(all_masks,mask)

        purple = [135,230,230]
        lower_purple = self.lower(purple,[3,80,40])
        upper_purple = self.upper(purple,[10,30,30])
        mask = cv2.inRange(hsv_image, lower_purple, upper_purple)
        all_masks = cv2.add(all_masks,mask)

        light_magenta = [158,255,255]
        lower_light_magenta = self.lower(light_magenta,[3,50,40])
        upper_light_magenta = self.upper(light_magenta,[3,10,20])
        mask = cv2.inRange(hsv_image, lower_light_magenta, upper_light_magenta)
        all_masks = cv2.add(all_masks,mask)


        all_colors = np.array([yellow_green,cyan_blue,purple,light_magenta])
        uppers = np.array([upper_yellow_green,upper_cyan_blue,upper_purple,upper_light_magenta])
        lowers = np.array([lower_yellow_green,lower_cyan_blue,lower_purple,lower_light_magenta])
        key_colors = np.array(["yellow_green","cyan_blue","purple","light_magenta"])

        # output = cv2.bitwise_and(cv_image, cv_image, mask = all_masks)

        # # kernel=np.ones((3,3),np.uint8)
        # # dilated=cv2.dilate(output,kernel,iterations=3)
        # gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # apply binary thresholding
        img2 = all_masks.copy()
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(all_masks, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        for i in range(0, nb_components):
            if sizes[i] <= 20:
                img2[output == i + 1] = 0
        # img2 = cv2.blur(img2, (3, 3))
        # ret, thresh = cv2.threshold(img2, 25, 255, cv2.THRESH_BINARY)
        # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        # visualize the binary image
        # cv2.imshow('Binary image', img2)
        # cv2.waitKey(1)

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        params.blobColor = 255

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 256

        # # Filter by Area.
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 10000000000

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
                color_name = self.which_color(hsv_image[cy_int,cx_int,:],all_colors,uppers,lowers,key_colors)
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
        #         msg = "({}, {})".format(cx_int,cy_int)
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
        return pixel_locations