
import numpy as np
import cv2

class LightDetector:
    def __init__(self):
        pass
    
    def upper (self, rgb):
        if (rgb.count(255) == 2 or rgb.count(0) == 2):
            adjust_brightness = True
        else: 
            adjust_brightness = False
        
        new_rgb = rgb.copy()
        for i in range(3):
            if new_rgb[i] == 0 and adjust_brightness:
                new_rgb[i] += 150
            else:
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
        result = np.isclose(rgb,all_colors,atol=75)
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

        # cyan = [252,255,40]
        cyan = [255,255,0]
        lower_cyan = self.lower(cyan)
        upper_cyan = self.upper(cyan)
        mask = cv2.inRange(cv_image, lower_cyan, upper_cyan)
        all_masks = mask

        # magenta = [234,37,245]
        magenta = [255,0,255]
        lower_magenta = self.lower(magenta)
        upper_magenta= self.upper(magenta)
        mask = cv2.inRange(cv_image, lower_magenta, upper_magenta)
        all_masks = cv2.add(all_masks,mask)

        # yellow = [45,253,250]
        yellow = [0,255,255]
        lower_yellow = self.lower(yellow)
        upper_yellow = self.upper(yellow)
        mask = cv2.inRange(cv_image, lower_yellow, upper_yellow)
        all_masks = cv2.add(all_masks,mask)
        
        red = [0,0,255]
        lower_red = self.lower(red)
        upper_red = self.upper(red)
        mask = cv2.inRange(cv_image, lower_red, upper_red)
        all_masks = cv2.add(all_masks,mask)
        
        # green = [25,255,0]
        green = [0,255,0]
        lower_green = self.lower(green)
        upper_green = self.upper(green)
        mask = cv2.inRange(cv_image, lower_green, upper_green)
        all_masks = cv2.add(all_masks,mask)
        
        # blue = [238,71,3]
        blue = [255,0,0]
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
            if sizes[i] <= 10:
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
        print(len(contours))
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

        # cv2.imwrite(os.path.normpath("points_detected" + '.png'), cv_image)
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