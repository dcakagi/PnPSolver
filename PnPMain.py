import numpy as np
from PoseEstimator import PoseEstimator

if __name__ == '__main__':

    constellation = {"cyan": [818.195312, -971.097656, 458.900879], "magenta": [867.039062, -11.128906, 674.893066],
                     "yellow": [867.039062, 698.871094, 524.893066], "red": [767.039062, 308.871094, 164.893066],
                     "green": [17.039062, -631.128906, 634.893066], "blue": [231.492188, 770.839844, 155.332031],
                     "red_magenta": [-664.863281, -416.613281, 423.383301], "orange": [-1042.960938, -1351.128906, 964.893066],
                     "blue_green": [-1052.960938, 78.871094, 584.893066]}

    pixels_height = 2160
    pixels_width = 3840
    fov = np.pi / 2
    focal_pixels = pixels_width / (2 * np.tan(fov / 2))

    estimator = PoseEstimator(constellation, pixels_width, pixels_height, focal_pixels)
    estimator_cv = PoseEstimator(constellation, pixels_width, pixels_height, focal_pixels, solver_type="opencv")

    pixels = {"blue_green": [2247, 1709], "orange": [1677, 1557], "red_magenta": [2040, 1722], "blue": [2395, 1711],
              "green": [1956, 1578], "red": [2224, 1658], "yellow": [2331, 1547], "magenta": [2128, 1504],
              "cyan": [1854, 1568]}

    pixels2 = {"blue_green": [2194, 1639],
               "red_magenta": [2021, 1607],
               "orange": [1712, 1528],
               "blue": [2328, 1522],
               "red": [2185, 1445],
               "green": [1951, 1439],
               "cyan": [1861, 1373],
               "yellow": [2285, 1352],
               "magenta": [2106, 1318]}

    pc_vp_est = estimator.updatePose(pixels)
    pc_vp_est_cv = estimator_cv.updatePose(pixels)

    #Extract position estimates
    xc_vp_est = pc_vp_est[0, 0]
    yc_vp_est = pc_vp_est[1, 0]
    zc_vp_est = pc_vp_est[2, 0]
    xc_vp_est_cv = pc_vp_est_cv[0, 0]
    yc_vp_est_cv = pc_vp_est_cv[1, 0]
    zc_vp_est_cv = pc_vp_est_cv[2, 0]

    #Actual vehicle position
    x_val = -5870.316222894529
    y_val = -743.7304690000019
    z_val = 2219.7767863354493

    x_error = abs((xc_vp_est - x_val) / x_val) * 100
    y_error = abs((yc_vp_est - y_val) / y_val) * 100
    z_error = abs((zc_vp_est - z_val) / z_val) * 100
    x_error_cv = abs((xc_vp_est_cv - x_val) / x_val) * 100
    y_error_cv = abs((yc_vp_est_cv - y_val) / y_val) * 100
    z_error_cv = abs((zc_vp_est_cv - z_val) / z_val) * 100

    print(f"\nOPnP X estimate: {xc_vp_est}")
    print(f"OPnP Y estimate: {yc_vp_est}")
    print(f"OPnP Z estimate: {zc_vp_est}")

    print(f"\nOPnP X % error: {x_error}")
    print(f"OPnP Y % error: {y_error}")
    print(f"OPnP Z % error: {z_error}")

    print(f"\nOpencv X estimate: {xc_vp_est_cv}")
    print(f"Opencv Y estimate: {yc_vp_est_cv}")
    print(f"Opencv Z estimate: {zc_vp_est_cv}")

    print(f"\nOpencv X % error: {x_error_cv}")
    print(f"Opencv Y % error: {y_error_cv}")
    print(f"Opencv Z % error: {z_error_cv}")

    pc_vp_est2 = estimator.updatePose(pixels2)
    pc_vp_est_cv2 = estimator_cv.updatePose(pixels2)

    #Extract position estimate
    xc_vp_est2 = pc_vp_est2[0, 0]
    yc_vp_est2 = pc_vp_est2[1, 0]
    zc_vp_est2 = pc_vp_est2[2, 0]
    xc_vp_est_cv2 = pc_vp_est_cv2[0, 0]
    yc_vp_est_cv2 = pc_vp_est_cv2[1, 0]
    zc_vp_est_cv2 = pc_vp_est_cv2[2, 0]

    x_val2 = -5886.406615960936
    y_val2 = -743.7304690000019
    z_val2 = 4184.527266987305

    x_error2 = abs((xc_vp_est2 - x_val2) / x_val2) * 100
    y_error2 = abs((yc_vp_est2 - y_val2) / y_val2) * 100
    z_error2 = abs((zc_vp_est2 - z_val2) / z_val2) * 100
    x_error_cv2 = abs((xc_vp_est_cv2 - x_val2) / x_val2) * 100
    y_error_cv2 = abs((yc_vp_est_cv2 - y_val2) / y_val2) * 100
    z_error_cv2 = abs((zc_vp_est_cv2 - z_val2) / z_val2) * 100

    print(f"\nX estimate: {xc_vp_est2}")
    print(f"Y estimate: {yc_vp_est2}")
    print(f"Z estimate: {zc_vp_est2}")

    print(f"\nX % error: {x_error2}")
    print(f"Y % error: {y_error2}")
    print(f"Z % error: {z_error2}")

    print(f"\nOpencv X estimate: {xc_vp_est_cv}")
    print(f"Opencv Y estimate: {yc_vp_est_cv}")
    print(f"Opencv Z estimate: {zc_vp_est_cv}")

    print(f"\nOpencv X % error: {x_error_cv}")
    print(f"Opencv Y % error: {y_error_cv}")
    print(f"Opencv Z % error: {z_error_cv}")