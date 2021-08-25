import numpy as np
import sys
import cv2

from light_detector import LightDetector

light_detector = LightDetector()

cv_image = cv2.imread("/home/brandon/projects/PnPSolver/simulation/image_3.png")

# cv2.imshow("image", cv_image)
# cv2.waitKey(1)

pixel_locations = light_detector.detect(cv_image,"front")

print(pixel_locations)