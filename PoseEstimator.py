import numpy as np
import cv2
import matlab.engine

class PoseEstimator:
    def __init__(self, constellation_: dict, pix_width: int, pix_height: int, focal_pix: float, solver_type: str="OPnP", method=cv2.SOLVEPNP_ITERATIVE, refinement='lm', use_guess=False):
        '''
        :param constellation_: dictionary of points with unique identifiers as keys and lists of (x, y, z) coordinates as values
        :param pix_width: pixels in image width
        :param pix_height: pixels in image height
        :param focal_pix: focal length of camera
        :param solver_type: 'OPnP', 'opencv', or 'opencv-ransac', type of solver used for estimation
        :param method: Method of PnP solving from opencv (if selected as solver_type) to use for pose estimation, see enumerations at
        https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gga357634492a94efe8858d0ce1509da869a9f589872a7f7d687dc58294e01ea33a5
        :param refinement: 'lm' or 'vvs', sets type of refinement method to use when calling refineEstimate()
        :param use_guess: bool determining if the previous location estimate will be used to initialize the iterative solution of the next pose estimate
        '''

        self.constellation = constellation_
        self.pixel_width = pix_width
        self.pixel_height = pix_height
        self.focal_length_pixels = focal_pix
        self.solver_type = solver_type
        self.location_estimate = None
        self.last_rotation = None
        self.last_translation = None
        self.flag = None
        self.cam_mat = None
        self.distortion = None
        self.refinement = refinement
        self.method = method
        self.use_guess = use_guess

        if solver_type == "OPnP":
            self.eng = matlab.engine.start_matlab()
            self.eng.addpath('OPnP')

    def updatePose(self, pixels: dict):
        '''
        :param pixels: dictionary of pixel locations with identifiers as keys and lists of (u, v) pixel coordinates as values
        NOTE: Identifiers for pixel coordinates must match those of constellation points
        '''

        pixel_points = []
        constellation_points = []
        for id in pixels:
            if id in self.constellation:
                pixel_points.append(pixels[id])
                constellation_points.append(self.constellation[id])

        pixel_points = np.array(pixel_points).astype(float)
        constellation_points = np.array(constellation_points).astype(float)

        if self.solver_type == 'OPnP':
            pixel_points = pixel_points.T
            constellation_points = constellation_points.T
            pixel_points[0, :] -= 1.0 * self.pixel_width / 2.0
            pixel_points[1, :] -= 1.0 * self.pixel_height / 2.0
            pixel_points[1, :] *= -1.0
            pixel_points = pixel_points / self.focal_length_pixels

            u = matlab.double(pixel_points.tolist())
            U = matlab.double(constellation_points.tolist())
            polish = 'polish'

            [R0, t0, error0, flag] = self.eng.OPnP(U, u, polish, nargout=4)

            self.last_rotation = np.array(R0)
            self.last_translation = np.array(t0)
            self.flag = flag
        elif self.solver_type == 'opencv' or self.solver_type == 'opencv-ransac':
            constellation_points = np.expand_dims(constellation_points, axis=2)
            pixel_points = np.expand_dims(pixel_points, axis=2)
            self.cam_mat = np.array([[self.focal_length_pixels, 0, self.pixel_width / 2],
                                [0, self.focal_length_pixels, self.pixel_height / 2],
                                [0, 0, 1]])
            self.distortion = None
            if self.solver_type == 'opencv':
                if self.use_guess and self.last_translation is not None:
                    prev_rot = cv2.Rodrigues(self.last_rotation)[0]
                    prev_trans = self.last_translation
                    ret, rvecs, tvecs = cv2.solvePnP(constellation_points, pixel_points, self.cam_mat, self.distortion, flags=self.method, useExtrinsicGuess=True, rvec=prev_rot, tvec=prev_trans)
                else:
                    ret, rvecs, tvecs = cv2.solvePnP(constellation_points, pixel_points, self.cam_mat, self.distortion, flags=self.method)
            else:  # For RANSAC
                if self.use_guess and self.last_translation is not None:
                    prev_rot = cv2.Rodrigues(self.last_rotation)[0]
                    prev_trans = self.last_translation
                    ret, rvecs, tvecs = cv2.solvePnPRansac(constellation_points, pixel_points, self.cam_mat, self.distortion, flags=self.method, useExtrinsicGuess=True, rvec=prev_rot, tvec=prev_trans)
                else:
                    ret, rvecs, tvecs = cv2.solvePnPRansac(constellation_points, pixel_points, self.cam_mat, self.distortion, flags=self.method)
            self.last_rotation = cv2.Rodrigues(rvecs)[0]
            self.last_translation = tvecs
            #print(ret)
            self.retval = ret

        self.location_estimate = -self.last_rotation.T @ self.last_translation

        return self.location_estimate

    def refineEstimate(self, pixels: dict):
        '''
        Refines an initial estimate using the same pixels that were used to calculate the initial estimate of the location
        :param pixels: dictionary of pixels, the same pixels used when calling updatePose() immediately prior
        :return: refined location estimate from the initial guess
        '''
        pixel_points = []
        constellation_points = []
        for id in pixels:
            if id in self.constellation:
                pixel_points.append(pixels[id])
                constellation_points.append(self.constellation[id])

        pixel_points = np.array(pixel_points).astype(float)
        pixel_points = np.expand_dims(pixel_points, axis=2)
        constellation_points = np.array(constellation_points).astype(float)
        constellation_points = np.expand_dims(constellation_points, axis=2)

        rotation = cv2.Rodrigues(self.last_rotation)[0]
        if self.refinement == 'lm':
            rvecs, tvecs = cv2.solvePnPRefineLM(constellation_points, pixel_points, self.cam_mat, self.distortion, rotation, self.last_translation)
        elif self.refinement == 'vvs':
            rvecs, tvecs = cv2.solvePnPRefineVVS(constellation_points, pixel_points, self.cam_mat, self.distortion, rotation, self.last_translation)

        self.last_rotation = cv2.Rodrigues(rvecs)[0]
        self.last_translation = tvecs
        self.location_estimate = -self.last_rotation.T @ self.last_translation

        return self.location_estimate

    def getLocationEstimate(self):
        return self.location_estimate

    def getRotationEstimate(self):
        return self.last_rotation

    def getTranslationEstimate(self):
        return self.last_translation

    def getFlag(self):
        return self.flag

    def setFocalLengthPixels(self, new_focal_length: float):
        self.focal_length_pixels = new_focal_length

    def setFocalLengthPixelsFromFOV(self, fov: float):
        self.focal_length_pixels = self.pixel_width / (2 * np.tan(fov / 2))
