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

    pixels = {"blue_green": [2247, 1709], "orange": [1677, 1557], "red_magenta": [2040, 1722], "blue": [2395, 1711],
              "green": [1956, 1578], "red": [2224, 1658], "yellow": [2331, 1547], "magenta": [2128, 1504],
              "cyan": [1854, 1568]}

    estimator.updatePose(pixels)
    rot = estimator.getRotationEstimate()
    trans = estimator.getTranlsationEstimate()
    rot = np.array(rot)
    trans = np.array(trans)
    print(rot)
    print(trans)

    ell_v_unit_est = rot @ np.array([[1, 0, 0]]).T
    el_est = np.arctan2(ell_v_unit_est[1, 0], ell_v_unit_est[2, 0])
    tmp = np.sqrt(ell_v_unit_est[1:2, 0].T @ ell_v_unit_est[1:2, 0])
    az_est = np.arctan2(ell_v_unit_est[0, 0], tmp)

    #Extract position estimate
    pc_vp_est = -rot.T @ trans
    xc_vp_est = pc_vp_est[0, 0]
    yc_vp_est = pc_vp_est[1, 0]
    zc_vp_est = pc_vp_est[2, 0]

    #Actual vehicle position
    x_val = -5870.316222894529
    y_val = -743.7304690000019
    z_val = -1724.2230763354492

    x_error = abs((xc_vp_est - x_val) / x_val) * 100
    y_error = abs((yc_vp_est - y_val) / y_val) * 100
    z_error = abs((zc_vp_est - z_val) / z_val) * 100

    print(f"X estimate: {xc_vp_est}")
    print(f"Y estimate: {yc_vp_est}")
    print(f"Z estimate: {zc_vp_est}")

    print(f"X % error: {x_error}")
    print(f"Y % error: {y_error}")
    print(f"Z % error: {z_error}")
