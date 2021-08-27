import numpy as np


def project_points(xc_vp, yc_vp, zc_vp, constellation, focal_pix, pix_width, pix_height, cam_elevation, cam_azumith):

    # Find unit vector pointing from camera to vertiport defined in body-fixed
    # inertial frame -- call it the vehicle frame (v)
    ell_v = np.array([[0 - xc_vp, 0 - yc_vp, 0 - zc_vp]]).T
    ell_v_unit = ell_v / np.linalg.norm(ell_v)

    # Define imaginary gimbal frame with x-axis pointing from gimbal/camera
    # position to vertiport.
    # azimuth is rotation from NED (vehicle) frame to gimbal-1 frame about
    # z-axis
    # elevation is rotation from gimbal-1 frame to gimbal frame about y-axis

    # Find azimuth and elevation angle to vertiport from camera location
    az = np.arctan2(ell_v_unit[1, 0], ell_v_unit[0, 0])
    el = -np.arcsin(ell_v_unit[2, 0])

    az = cam_azumith
    el = cam_elevation

    # Find rotations between vehicle frame and camera frame
    # See sections 13.1 and 13.2 in uavbook
    R_v_g1 = np.array([[np.cos(az), np.sin(az), 0],
                       [-np.sin(az), np.cos(az), 0],
                       [0, 0, 1]])
    R_g1_g = np.array([[np.cos(el), 0, -np.sin(el)],
                       [0, 1, 0],
                       [np.sin(el), 0, np.cos(el)]])
    R_v_g = R_g1_g @ R_v_g1
    R_g_v = R_v_g.T

    R_g_c = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0]])
    R_c_g = R_g_c.T

    R_c_v = R_g_v @ R_c_g  # Camera to vehicle rotation
    R_v_c = R_c_v.T

    xp_vp = constellation[0, :]
    yp_vp = constellation[1, :]
    zp_vp = constellation[2, :]

    # Calculate pointing vectors from camera to point light sources in vehicle
    # (same as vertiport, both NED) frame
    ell_pts_v = np.squeeze(np.array([xp_vp - xc_vp, yp_vp - yc_vp, zp_vp - zc_vp]))
    # Express in camera frame
    ell_pts_c = R_v_c @ ell_pts_v

    # Camera-frame z-distance to light sources
    lamb = ell_pts_c[2, :]

    # light source pixel locations, float, no noise -- perfect imaging
    fpx = focal_pix
    xp_c = fpx / lamb * ell_pts_c[0, :]
    yp_c = fpx / lamb * ell_pts_c[1, :]

    # Round to nearest integer pixel values
    xp_c = np.round(xp_c) + pix_width / 2
    yp_c = np.round(yp_c) + pix_height / 2

    # For float pixel values
    # xp_c = xp_c + pix_width / 2.0
    # yp_c = yp_c + pix_height / 2.0

    uu = np.array([xp_c, yp_c])

    return uu

if __name__ == '__main__':
    from PoseEstimator import PoseEstimator

    constellation_points = 20
    horiz_dim = 40
    vert_dim = 40

    camera_elevation = 7.0 * np.pi / 180  # 7 degree tilted camera
    camera_azumith = np.pi  # Camera facing straight forward (in vehicle frame)

    x_pos = 100
    y_pos = 100
    z_pos = -50

    constellation = np.zeros((3, constellation_points))
    constellation[0:2, :] = np.random.uniform(-horiz_dim/2, horiz_dim/2, (2, constellation_points))
    constellation[2, :] = np.random.uniform(0, vert_dim, (1, constellation_points))

    constellation_dict = {}
    for m in range(constellation_points):
        constellation_dict[m] = constellation[:, m]

    pix_width = 3840
    pix_height = 2160
    fov = np.pi / 2
    focal_pix = pix_width / (2 * np.tan(fov / 2))

    pix_locations = project_points(x_pos, y_pos, z_pos, constellation, focal_pix, pix_width, pix_height, camera_elevation, camera_azumith)

    pixels = {}
    for m in range(constellation_points):
        pixels[m] = pix_locations[:, m]

    estimator = PoseEstimator(constellation_dict, pix_width, pix_height, focal_pix, solver_type="opencv")

    est_pose = estimator.updatePose(pixels)

    print(est_pose)
