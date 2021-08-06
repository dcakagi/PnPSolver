import numpy as np
import matlab.engine

class PoseEstimator:
    def __init__(self, constellation_: dict, pix_width: int, pix_height: int, flength_pix: float):
        '''
        :param constellation_: dictionary of points with unique identifiers as keys and (x, y, z) coordinates as values
        :param pix_width: pixels in image width
        :param pix_height: pixels in image height
        :param flength_pix: focal length of camera
        '''

        self.constellation = constellation_
        self.pixel_width = pix_width
        self.pixel_height = pix_height
        self.focal_length_pixels = flength_pix
        self.last_rotation = None
        self.last_translation = None
        self.flag = None

        self.eng = matlab.engine.start_matlab()

    def updatePose(self, pixels: dict):
        '''
        :param pixels: dictionary of pixel locations with identifiers as keys and (u, v) pixel coordinates as values
        NOTE: Identifiers for pixel coordinates must match those of constellation points
        '''

        pixel_points = []
        constellation_points = []
        for id in pixels:
            if id in self.constellation:
                pixel_points.append(pixels[id])
                constellation_points.append(self.constellation[id])

        pixel_points = np.array(pixel_points).T.astype(np.float)
        constellation_points = np.array(constellation_points).T.astype(np.float)

        pixel_points[0, :] -= 1.0 * self.pixel_width / 2.0
        pixel_points[1, :] -= 1.0 * self.pixel_height / 2.0
        pixel_points[1, :] *= -1.0
        pixel_points = pixel_points / self.focal_length_pixels

        u = matlab.double(pixel_points.tolist())
        U = matlab.double(constellation_points.tolist())
        polish = 'polish'

        [R0, t0, error0, flag] = self.eng.OPnP(U, u, polish, nargout=4)

        self.last_rotation = R0
        self.last_translation = t0
        self.flag = flag

    def getRotationEstimate(self):
        return self.last_rotation

    def getTranlsationEstimate(self):
        return self.last_translation

    def getFlag(self):
        return self.flag

