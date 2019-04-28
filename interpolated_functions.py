"""
This file contains the unidimensionalized and the interpolated coupling
and constraint outputs for the SSBJ problem as shown in the vanaret paper
"""

# Imports
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pylab as plt
from ssbjkadmos.utils.math import polynomial_function
import random
import pickle

# Initialize bounds for component outputs
bounds_dict = pickle.load(open("bounds_dict.p", "rb"))


class StructureInterpolation:

    def __init__(self):
        self.scale = np.array([1 * 1.2, .45 * .98])
        #  :lower and upper bounds on design and coupling variables
        self.x_str = np.array([[0.4, 0.6], [.75, 1.25]])  # : lambda_max raises theta initial
        self.z = np.array([[0.034, 0.14], [20000, 45000], [.5, 1.2], [3.0, 8], [30, 190], [600, 2100]]) * self.scale
        #  :z[3][0] changes convexity of WFW & Theta; z[4] alters WT; z[5]_max changes Theta smoothness
        self.load = np.array([15000, 70000]) * self.scale
        self.we = np.array([0, 70000]) * self.scale
        self.nz = 6.0
        self.wfo = 2000.0
        self.wo = 25000.0

    @staticmethod
    def y12(x_str, z, load, w_e, n_z, w_fo, w_o, component):

        # common calculations
        t = z[0] * z[5] / (np.sqrt(abs(z[5] * z[3])))
        b = np.sqrt(abs(z[5] * z[3])) / 2.0
        r = (1.0 + 2.0 * x_str[0]) / (3.0 * (1.0 + x_str[0]))

        # calculations for specific components
        if component == 0:
            fo1 = polynomial_function([x_str[1]], [1], [.008], "Fo1")
            wt_hat = load
            ww = fo1 * (0.0051 * abs(wt_hat * n_z) ** 0.557 *
                        abs(z[5]) ** 0.649 * abs(z[3]) ** 0.5 * abs(z[0]) ** (-0.4)
                        * abs(1.0 + x_str[0]) ** 0.1 * (0.1875 * abs(z[5])) ** 0.1
                        / abs(np.cos(z[4] * np.pi / 180.)))
            wfw = 5.0 / 18.0 * abs(z[5]) * 2.0 / 3.0 * t * 42.5
            wf = wfw + w_fo
            wt = w_o + ww + wf + w_e

            return wt

        if component == 1:
            theta = polynomial_function([abs(x_str[1]), b, r, load],
                                        [2, 4, 4, 3], [0.25] * 4, "twist")

            return theta

    def y12_int(self, diagonal, component):

        scale = self.scale
        x_str = self.x_str
        z = self.z
        load = self.load
        we = np.array([0, 70000]) * scale
        nz = 6.0
        wfo = 2000.0
        wo = 25000.0

        # calculation of interpolated values for each component
        if component == 0:
            min_y_12 = bounds_dict['y12_min'][0]
            max_y_12 = bounds_dict['y12_max'][0]

            y_12 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                we_0 = we[0] + t * (we[1] - we[0])
                wt = self.y12([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 0)
                y_12.append(wt)
            y_12 = np.asarray(y_12)
            y_12 = (y_12 - min_y_12) / (max_y_12 - min_y_12)

            return y_12

        if component == 1:
            min_y_12 = bounds_dict['y12_min'][1]
            max_y_12 = bounds_dict['y12_max'][1]

            y_12 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                we_0 = we[0] + t * (we[1] - we[0])
                theta = self.y12([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 1)
                y_12.append(theta)
            y_12 = np.asarray(y_12)
            y_12 = (y_12 - min_y_12) / (max_y_12 - min_y_12)

            return y_12

    @staticmethod
    def y14(x_str, z, load, w_e, n_z, w_fo, w_o, component):

        # common calculations
        t = z[0] * z[5] / (np.sqrt(abs(z[5] * z[3])))

        # calculations for specific components
        if component == 0:
            fo1 = polynomial_function([x_str[1]], [1], [.008], "Fo1")
            wt_hat = load
            ww = fo1 * (0.0051 * abs(wt_hat * n_z) ** 0.557 *
                        abs(z[5]) ** 0.649 * abs(z[3]) ** 0.5 * abs(z[0]) ** (-0.4)
                        * abs(1.0 + x_str[0]) ** 0.1 * (0.1875 * abs(z[5])) ** 0.1
                        / abs(np.cos(z[4] * np.pi / 180.)))
            wfw = 5.0 / 18.0 * abs(z[5]) * 2.0 / 3.0 * t * 42.5
            wf = wfw + w_fo
            wt = w_o + ww + wf + w_e

            return wt

        if component == 1:
            wfw = 5.0 / 18.0 * abs(z[5]) * 2.0 / 3.0 * t * 42.5
            wf = wfw + w_fo

            return wf

    def y14_int(self, diagonal, component):

        scale = self.scale
        x_str = self.x_str
        z = self.z
        load = self.load
        we = np.array([0, 70000]) * scale
        nz = 6.0
        wfo = 2000.0
        wo = 25000.0

        # calculation of interpolated values for each component
        if component == 0:
            min_y_14 = bounds_dict['y14_min'][0]
            max_y_14 = bounds_dict['y14_max'][0]

            y_14 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                we_0 = we[0] + t * (we[1] - we[0])
                wt = self.y14([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 0)
                y_14.append(wt)
            y_14 = np.asarray(y_14)
            y_14 = (y_14 - min_y_14) / (max_y_14 - min_y_14)

            return y_14

        if component == 1:
            min_y_14 = bounds_dict['y14_min'][1]
            max_y_14 = bounds_dict['y14_max'][1]

            y_14 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                we_0 = we[0] + t * (we[1] - we[0])
                wf = self.y14([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 1)
                y_14.append(wf)
            y_14 = np.asarray(y_14)
            y_14 = (y_14 - min_y_14) / (max_y_14 - min_y_14)
            return y_14

    @staticmethod
    def y11(z, w_fo, component):

        # common calculations
        t = z[0] * z[5] / (np.sqrt(abs(z[5] * z[3])))

        # calculations for specific components
        if component == 0:
            wfw = 5.0 / 18.0 * abs(z[5]) * 2.0 / 3.0 * t * 42.5
            wf = wfw + w_fo

            return wf

    def y11_int(self, diagonal, component):

        z = self.z
        wfo = 2000.0

        # calculation of interpolated values for each component
        if component == 0:
            min_y_11 = bounds_dict['y11_min'][0]
            max_y_11 = bounds_dict['y11_max'][0]

            y_11 = []
            for t in diagonal:
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                wt = self.y11([z_0, z_1, z_2, z_3, z_4, z_5], wfo, 0)
                y_11.append(wt)
            y_11 = np.asarray(y_11)
            y_11 = (y_11 - min_y_11) / (max_y_11 - min_y_11)

            return y_11

    @staticmethod
    def y1(x_str, z, load, w_e, n_z, w_fo, w_o, component):

        # common calculations
        t = z[0] * z[5] / (np.sqrt(abs(z[5] * z[3])))

        # calculations for specific components
        if component == 0:
            fo1 = polynomial_function([x_str[1]], [1], [.008], "Fo1")
            wt_hat = load
            ww = fo1 * (0.0051 * abs(wt_hat * n_z) ** 0.557 *
                        abs(z[5]) ** 0.649 * abs(z[3]) ** 0.5 * abs(z[0]) ** (-0.4)
                        * abs(1.0 + x_str[0]) ** 0.1 * (0.1875 * abs(z[5])) ** 0.1
                        / abs(np.cos(z[4] * np.pi / 180.) + 0.01))
            wfw = 5.0 / 18.0 * abs(z[5]) * 2.0 / 3.0 * t * 42.5
            wf = wfw + w_fo
            wt = w_o + ww + wf + w_e

            return wt

        if component == 1:
            wfw = 5.0 / 18.0 * abs(z[5]) * 2.0 / 3.0 * t * 42.5

            return wfw

        if component == 2:
            b = np.sqrt(abs(z[5] * z[3])) / 2.0
            r = (1.0 + 2.0 * x_str[0]) / (3.0 * (1.0 + x_str[0]))
            theta = polynomial_function([abs(x_str[1]), b, r, load],
                                        [2, 4, 4, 3], [0.25] * 4, "twist")

            return theta

    def y1_int(self, diagonal, component):

        scale = self.scale
        x_str = self.x_str
        z = self.z
        load = self.load
        we = np.array([0, 70000]) * scale
        nz = 6.0
        wfo = 2000.0
        wo = 25000.0

        # calculation of interpolated values for each component
        if component == 0:
            min_y_1 = bounds_dict['y1_min'][0]
            max_y_1 = bounds_dict['y1_max'][0]

            y_1 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                we_0 = we[0] + t * (we[1] - we[0])
                wt = self.y1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 0)
                y_1.append(wt)
            y_1 = np.asarray(y_1)
            y_1 = (y_1 - min_y_1) / (max_y_1 - min_y_1)

            return y_1

        if component == 1:
            min_y_1 = bounds_dict['y1_min'][1]
            max_y_1 = bounds_dict['y1_max'][1]

            y_1 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                we_0 = we[0] + t * (we[1] - we[0])
                wt = self.y1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 1)
                y_1.append(wt)
            y_1 = np.asarray(y_1)
            y_1 = (y_1 - min_y_1) / (max_y_1 - min_y_1)

            return y_1

        if component == 2:
            min_y_1 = bounds_dict['y1_min'][2]
            max_y_1 = bounds_dict['y1_max'][2]

            y_1 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                we_0 = we[0] + t * (we[1] - we[0])
                wf = self.y1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 2)
                y_1.append(wf)
            y_1 = np.asarray(y_1)
            y_1 = (y_1 - min_y_1) / (max_y_1 - min_y_1)

            return y_1

    @staticmethod
    def g1(x_str, z, load, component):

        if component == 0:
            b = np.sqrt(abs(z[5] * z[3])) / 2.0
            r = (1.0 + 2.0 * x_str[0]) / (3.0 * (1.0 + x_str[0]))
            sigma = 5 * [0.]
            sigma[0] = polynomial_function([z[0], load, x_str[1], b, r], [4, 1, 4, 1, 1], [0.1] * 5, "sigma[1]")

            return sigma[0] - 1.09

        if component == 1:
            b = np.sqrt(abs(z[5] * z[3])) / 2.0
            r = (1.0 + 2.0 * x_str[0]) / (3.0 * (1.0 + x_str[0]))
            sigma = 5 * [0.]
            sigma[1] = polynomial_function([z[0], load, x_str[1], b, r], [4, 1, 4, 1, 1], [0.15] * 5, "sigma[2]")

            return sigma[1] - 1.09

        if component == 2:
            b = np.sqrt(abs(z[5] * z[3])) / 2.0
            r = (1.0 + 2.0 * x_str[0]) / (3.0 * (1.0 + x_str[0]))
            sigma = 5 * [0.]
            sigma[2] = polynomial_function([z[0], load, x_str[1], b, r], [4, 1, 4, 1, 1], [0.2] * 5, "sigma[3]")

            return sigma[2] - 1.09

        if component == 3:
            b = np.sqrt(abs(z[5] * z[3])) / 2.0
            r = (1.0 + 2.0 * x_str[0]) / (3.0 * (1.0 + x_str[0]))
            sigma = 5 * [0.]
            sigma[3] = polynomial_function([z[0], load, x_str[1], b, r], [4, 1, 4, 1, 1], [0.25] * 5, "sigma[4]")

            return sigma[3] - 1.09

        if component == 4:
            b = np.sqrt(abs(z[5] * z[3])) / 2.0
            r = (1.0 + 2.0 * x_str[0]) / (3.0 * (1.0 + x_str[0]))
            sigma = 5 * [0.]
            sigma[4] = polynomial_function([z[0], load, x_str[1], b, r], [4, 1, 4, 1, 1], [0.30] * 5, "sigma[5]")

            return sigma[4] - 1.09

        if component == 5:
            b = np.sqrt(abs(z[5] * z[3])) / 2.0
            r = (1.0 + 2.0 * x_str[0]) / (3.0 * (1.0 + x_str[0]))
            theta = polynomial_function([abs(x_str[1]), b, r, load], [2, 4, 4, 3], [0.25] * 4, "twist")

            return theta - 1.04

        if component == 6:
            b = np.sqrt(abs(z[5] * z[3])) / 2.0
            r = (1.0 + 2.0 * x_str[0]) / (3.0 * (1.0 + x_str[0]))
            theta = polynomial_function([abs(x_str[1]), b, r, load], [2, 4, 4, 3], [0.25] * 4, "twist")

            return 0.96 - theta

    def g1_int(self, diagonal, component):

        x_str = self.x_str
        z = self.z
        load = self.load

        if component == 0:
            min_g_1_1 = bounds_dict['g1_min'][0]
            max_g_1_1 = bounds_dict['g1_max'][0]

            g_1 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                sigma1 = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 0)
                g_1.append(sigma1)
                g_1 = np.asarray(g_1)
                g_1 = (g_1 - min_g_1_1) / (max_g_1_1 - min_g_1_1)

            return g_1

        if component == 1:
            min_g_1_2 = bounds_dict['g1_min'][1]
            max_g_1_2 = bounds_dict['g1_max'][1]

            g_1 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                sigma2 = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 1)
                g_1.append(sigma2)
                g_1 = np.asarray(g_1)
                g_1 = (g_1 - min_g_1_2) / (max_g_1_2 - min_g_1_2)

            return g_1

        if component == 2:
            min_g_1_3 = bounds_dict['g1_min'][2]
            max_g_1_3 = bounds_dict['g1_max'][2]

            g_1 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                sigma3 = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 2)
                g_1.append(sigma3)
                g_1 = np.asarray(g_1)
                g_1 = (g_1 - min_g_1_3) / (max_g_1_3 - min_g_1_3)

            return g_1

        if component == 3:
            min_g_1_4 = bounds_dict['g1_min'][3]
            max_g_1_4 = bounds_dict['g1_max'][3]

            g_1 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                sigma4 = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 3)
                g_1.append(sigma4)
                g_1 = np.asarray(g_1)
                g_1 = (g_1 - min_g_1_4) / (max_g_1_4 - min_g_1_4)

            return g_1

        if component == 4:
            min_g_1_5 = bounds_dict['g1_min'][4]
            max_g_1_5 = bounds_dict['g1_max'][4]

            g_1 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                sigma5 = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 4)
                g_1.append(sigma5)
                g_1 = np.asarray(g_1)
                g_1 = (g_1 - min_g_1_5) / (max_g_1_5 - min_g_1_5)

            return g_1

        if component == 5:
            min_g_1_6 = bounds_dict['g1_min'][5]
            max_g_1_6 = bounds_dict['g1_max'][5]

            g_1 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                theta = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 5)
                g_1.append(theta)
                g_1 = np.asarray(g_1)
                g_1 = (g_1 - min_g_1_6) / (max_g_1_6 - min_g_1_6)

            return g_1

        if component == 6:
            min_g_1_7 = bounds_dict['g1_min'][6]
            max_g_1_7 = bounds_dict['g1_max'][6]

            g_1 = []
            for t in diagonal:
                x_str_0 = x_str[0][0] + t * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                l_0 = load[0] + t * (load[1] - load[0])
                theta = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 6)
                g_1.append(theta)
                g_1 = np.asarray(g_1)
                g_1 = (g_1 - min_g_1_7) / (max_g_1_7 - min_g_1_7)

            return g_1


class AerodynamicsInterpolation:

    def __init__(self):
        self.z = StructureInterpolation().z
        self.scale = np.array([1 * 1.2, .45 * .98])
        self.x_aer = np.array([0.65, 1.25]) * self.scale
    # self.z = np.array([[0.034, 0.14], [30000, 120000], [.4, 1.45], [3.0, 8], [30, 190], [600, 2100]]) * self.scale1
        self.wt = np.array([15000., 70000.]) * self.scale
        self.esf = np.array([0.5, 1.5]) * self.scale
        self.theta = np.array([0.96, 1.04])
        self.cdmin = 0.01375

    @staticmethod
    def y21(w_t, component):

        if component == 0:
            load = w_t

            return load

    def y21_int(self, diagonal, component):
        wt = self.wt

        if component == 0:

            min_y_21 = bounds_dict['y21_min'][0]
            max_y_21 = bounds_dict['y21_max'][0]

            y_21 = []
            for t in diagonal:
                wt_0 = wt[0] + t * (wt[1] - wt[0])
                load = self.y21(wt_0, 0)
                y_21.append(load)

            y_21 = np.asarray(y_21)
            y_21 = (y_21 - min_y_21) / (max_y_21 - min_y_21)

            return y_21

    @staticmethod
    def y23(x_aer, z, w_t, esf, theta, cdmin, component):

        if component == 0:
            if z[1] <= 36089.0:
                v = 1116.39 * z[2] * np.sqrt(abs(1.0 - 6.875E-6 * z[1]))
                rho = 2.377E-3 * (1. - 6.875E-6 * z[1]) ** 4.2561
            else:
                v = 968.1 * abs(z[2])
                rho = 2.377E-3 * 0.2971 * np.exp((36089.0 - z[1]) / 20806.7)
            cl = w_t / (0.5 * rho * (v ** 2) * z[5] + .001)
            fo2 = polynomial_function([esf, abs(x_aer)], [1, 1], [.25] * 2, "Fo2")

            c_dmin = cdmin * fo2 + 3.05 * abs(z[0]) ** (5.0 / 3.0) * abs(np.cos(z[4] * np.pi / 180.0)) ** 1.5
            if z[2] >= 1:
                k = abs(z[3]) * (abs(z[2]) ** 2 - 1.0) * np.cos(z[4] * np.pi / 180.) \
                    / (4. * abs(z[3]) * np.sqrt(abs(z[4] ** 2 - 1.) - 2.))
            else:
                k = (0.8 * np.pi * abs(z[3])) ** -1

            fo3 = polynomial_function([theta], [5], [.25], "Fo3")
            cd = (c_dmin + k * cl ** 2) * fo3
            d = cd * 0.5 * rho * v ** 2 * z[5]

            return d

    def y23_int(self, diagonal, component):
        x_aer = self.x_aer
        z = self.z
        wt = self.wt
        esf = self.esf
        theta = self.theta
        cdmin = self.cdmin

        if component == 0:
            min_y_23 = bounds_dict['y23_min'][0]
            max_y_23 = bounds_dict['y23_max'][0]

            y_23 = []
            for t in diagonal:
                x_aer_0 = x_aer[0] + t * (x_aer[1] - x_aer[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                wt_0 = wt[0] + t * (wt[1] - wt[0])
                esf_0 = esf[0] + t * (esf[1] - esf[0])
                theta_0 = theta[0] + t * (theta[1] - theta[0])
                d = self.y23(x_aer_0, [z_0, z_1, z_2, z_3, z_4, z_5], wt_0, esf_0, theta_0, cdmin, 0)
                y_23.append(d)
            y_23 = np.asarray(y_23)
            y_23 = (y_23 - min_y_23) / (max_y_23 - min_y_23)

            return y_23

    @staticmethod
    def y24(x_aer, z, w_t, esf, theta, cdmin, component):

        if component == 0:
            if z[1] <= 36089.0:
                v = 1116.39 * z[2] * np.sqrt(abs(1.0 - 6.875E-6 * z[1]))
                rho = 2.377E-3 * (1. - 6.875E-6 * z[1]) ** 4.2561
            else:
                v = 968.1 * abs(z[2])
                rho = 2.377E-3 * 0.2971 * np.exp((36089.0 - z[1]) / 20806.7)
            cl = w_t / (0.5 * rho * (v ** 2) * z[5] + .001)
            fo2 = polynomial_function([esf, abs(x_aer)], [1, 1], [.25] * 2, "Fo2")

            c_dmin = cdmin * fo2 + 3.05 * abs(z[0]) ** (5.0 / 3.0) * abs(np.cos(z[4] * np.pi / 180.0)) ** 1.5
            if z[2] >= 1:
                k = abs(z[3]) * (abs(z[2]) ** 2 - 1.0) * np.cos(z[4] * np.pi / 180.) \
                    / (4. * abs(z[3]) * np.sqrt(abs(z[4] ** 2 - 1.) - 2.))
            else:
                k = (0.8 * np.pi * abs(z[3])) ** -1

            fo3 = polynomial_function([theta], [5], [.25], "Fo3")
            cd = (c_dmin + k * cl ** 2) * fo3
            d = cd * 0.5 * rho * v ** 2 * z[5]
            fin = w_t / (d + 0.01)

            return fin

    def y24_int(self, diagonal, component):
        x_aer = self.x_aer
        z = self.z
        wt = self.wt
        esf = self.esf
        theta = self.theta
        cdmin = self.cdmin

        if component == 0:

            min_y_24 = bounds_dict['y24_min'][0]
            max_y_24 = bounds_dict['y24_max'][0]

            y_24 = []
            for t in diagonal:
                x_aer_0 = x_aer[0] + t * (x_aer[1] - x_aer[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                wt_0 = wt[0] + t * (wt[1] - wt[0])
                esf_0 = esf[0] + t * (esf[1] - esf[0])
                theta_0 = theta[0] + t * (theta[1] - theta[0])
                fin = self.y24(x_aer_0, [z_0, z_1, z_2, z_3, z_4, z_5], wt_0, esf_0, theta_0, cdmin, 0)
                y_24.append(fin)
            y_24 = np.asarray(y_24)
            y_24 = (y_24 - min_y_24) / (max_y_24 - min_y_24)

            return y_24

    @staticmethod
    def y2(x_aer, z, w_t, esf, theta, cdmin, component):

        if component == 0:
            load = w_t

            return load

        if component == 1:
            if z[1] <= 36089.0:
                v = 1116.39 * z[2] * np.sqrt(abs(1.0 - 6.875E-6 * z[1]))
                rho = 2.377E-3 * (1. - 6.875E-6 * z[1]) ** 4.2561
            else:
                v = 968.1 * abs(z[2])
                rho = 2.377E-3 * 0.2971 * np.exp((36089.0 - z[1]) / 20806.7)
            cl = w_t / (0.5 * rho * (v ** 2) * z[5] + .001)
            fo2 = polynomial_function([esf, abs(x_aer)], [1, 1], [.25] * 2, "Fo2")

            c_dmin = cdmin * fo2 + 3.05 * abs(z[0]) ** (5.0 / 3.0) * abs(np.cos(z[4] * np.pi / 180.0)) ** 1.5
            if z[2] >= 1:
                k = abs(z[3]) * (abs(z[2]) ** 2 - 1.0) * np.cos(z[4] * np.pi / 180.) \
                    / (4. * abs(z[3]) * np.sqrt(abs(z[4] ** 2 - 1.) - 2.))
            else:
                k = (0.8 * np.pi * abs(z[3])) ** -1

            fo3 = polynomial_function([theta], [5], [.25], "Fo3")
            cd = (c_dmin + k * cl ** 2) * fo3
            d = cd * 0.5 * rho * v ** 2 * z[5]

            return d

        if component == 2:
            if z[1] <= 36089.0:
                v = 1116.39 * z[2] * np.sqrt(abs(1.0 - 6.875E-6 * z[1]))
                rho = 2.377E-3 * (1. - 6.875E-6 * z[1]) ** 4.2561
            else:
                v = 968.1 * abs(z[2])
                rho = 2.377E-3 * 0.2971 * np.exp((36089.0 - z[1]) / 20806.7)
            cl = w_t / (0.5 * rho * (v ** 2) * z[5] + .001)
            fo2 = polynomial_function([esf, abs(x_aer)], [1, 1], [.25] * 2, "Fo2")

            c_dmin = cdmin * fo2 + 3.05 * abs(z[0]) ** (5.0 / 3.0) * abs(np.cos(z[4] * np.pi / 180.0)) ** 1.5
            if z[2] >= 1:
                k = abs(z[3]) * (abs(z[2]) ** 2 - 1.0) * np.cos(z[4] * np.pi / 180.) \
                    / (4. * abs(z[3]) * np.sqrt(abs(z[4] ** 2 - 1.) - 2.))
            else:
                k = (0.8 * np.pi * abs(z[3])) ** -1

            fo3 = polynomial_function([theta], [5], [.25], "Fo3")
            cd = (c_dmin + k * cl ** 2) * fo3
            d = cd * 0.5 * rho * v ** 2 * z[5]
            fin = w_t / (d + 0.01)

            return fin

    def y2_int(self, diagonal, component):

        x_aer = self.x_aer
        z = self.z
        wt = self.wt
        esf = self.esf
        theta = self.theta
        cdmin = self.cdmin

        if component == 0:
            y_2 = []
            for a1 in range(10):
                t1 = np.linspace(0, 1, 10)[a1]
                x_aer_0 = x_aer[0] + t1 * (x_aer[1] - x_aer[0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                wt_0 = wt[0] + t1 * (wt[1] - wt[0])
                esf_0 = esf[0] + t1 * (esf[1] - esf[0])
                theta_0 = theta[0] + t1 * (theta[1] - theta[0])
                load = self.y2(x_aer_0, [z_0, z_1, z_2, z_3, z_4, z_5], wt_0, esf_0, theta_0, cdmin, 0)
                y_2.append(load)

            min_y_2 = min(y_2)
            max_y_2 = max(y_2)

            y_2 = []
            for t in diagonal:
                x_aer_0 = x_aer[0] + t * (x_aer[1] - x_aer[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                wt_0 = wt[0] + t * (wt[1] - wt[0])
                esf_0 = esf[0] + t * (esf[1] - esf[0])
                theta_0 = theta[0] + t * (theta[1] - theta[0])
                load = self.y2(x_aer_0, [z_0, z_1, z_2, z_3, z_4, z_5], wt_0, esf_0, theta_0, cdmin, 0)
                y_2.append(load)

            y_2 = np.asarray(y_2)
            y_2 = (y_2 - min_y_2) / (max_y_2 - min_y_2)

            return y_2

        if component == 1:
            y_2 = []
            for a1 in range(10):
                t1 = np.linspace(0, 1, 10)[a1]
                x_aer_0 = x_aer[0] + t1 * (x_aer[1] - x_aer[0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                wt_0 = wt[0] + t1 * (wt[1] - wt[0])
                esf_0 = esf[0] + t1 * (esf[1] - esf[0])
                theta_0 = theta[0] + t1 * (theta[1] - theta[0])
                drag = self.y2(x_aer_0, [z_0, z_1, z_2, z_3, z_4, z_5], wt_0, esf_0, theta_0, cdmin, 1)
                y_2.append(drag)

            min_y_2 = min(y_2)
            max_y_2 = max(y_2)

            y_2 = []
            for t in diagonal:
                x_aer_0 = x_aer[0] + t * (x_aer[1] - x_aer[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                wt_0 = wt[0] + t * (wt[1] - wt[0])
                esf_0 = esf[0] + t * (esf[1] - esf[0])
                theta_0 = theta[0] + t * (theta[1] - theta[0])
                drag = self.y2(x_aer_0, [z_0, z_1, z_2, z_3, z_4, z_5], wt_0, esf_0, theta_0, cdmin, 1)
                y_2.append(drag)

            y_2 = np.asarray(y_2)
            y_2 = (y_2 - min_y_2) / (max_y_2 - min_y_2)

            return y_2

        if component == 2:
            y_2 = []
            for a1 in range(10):
                t1 = np.linspace(0, 1, 10)[a1]
                x_aer_0 = x_aer[0] + t1 * (x_aer[1] - x_aer[0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                wt_0 = wt[0] + t1 * (wt[1] - wt[0])
                esf_0 = esf[0] + t1 * (esf[1] - esf[0])
                theta_0 = theta[0] + t1 * (theta[1] - theta[0])
                fin = self.y2(x_aer_0, [z_0, z_1, z_2, z_3, z_4, z_5], wt_0, esf_0, theta_0, cdmin, 2)
                y_2.append(fin)

            min_y_2 = min(y_2)
            max_y_2 = max(y_2)

            y_2 = []
            for t in diagonal:
                x_aer_0 = x_aer[0] + t * (x_aer[1] - x_aer[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                wt_0 = wt[0] + t * (wt[1] - wt[0])
                esf_0 = esf[0] + t * (esf[1] - esf[0])
                theta_0 = theta[0] + t * (theta[1] - theta[0])
                fin = self.y2(x_aer_0, [z_0, z_1, z_2, z_3, z_4, z_5], wt_0, esf_0, theta_0, cdmin, 2)
                y_2.append(fin)

            y_2 = np.asarray(y_2)
            y_2 = (y_2 - min_y_2) / (max_y_2 - min_y_2)

            return y_2

    @staticmethod
    def g2(z, component):

        if component == 0:
            dpdx = polynomial_function([z[0]], [1], [.25], "dpdx")

            return dpdx - 1.04

    def g2_int(self, diagonal, component):
        z = self.z

        if component == 0:
            g_2 = []
            for a1 in range(10):
                t1 = np.linspace(0, 1, 10)[a1]
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                dpdx = self.g2([z_0, z_1, z_2, z_3, z_4, z_5], 0)
                g_2.append(dpdx)

            min_g_2 = min(g_2)
            max_g_2 = max(g_2)

            g_2 = []
            for t in diagonal:
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                dpdx = self.g2([z_0, z_1, z_2, z_3, z_4, z_5], 0)
                g_2.append(dpdx)

            g_2 = np.asarray(g_2)
            g_2 = (g_2 - min_g_2) / (max_g_2 - min_g_2)

            return g_2


class PropulsionInterpolation:

    def __init__(self):
        self.z = StructureInterpolation().z
        self.scale1 = np.array([1 * 1.2, .45 * .98])
        self.x_pro = np.array([0.37, .6])
        self.d = np.array([1000., 15000.])
        self.wbe = 4360

    @staticmethod
    def y3(x_pro, z, drag, component):

        if component == 0:
            temp = polynomial_function([z[2], z[1], abs(x_pro)], [2, 4, 2], [.25] * 3, "Temp")

            return temp

        if component in [1, 2]:
            tbar = abs(x_pro) * 16168.6
            esf = (drag / 3.0) / tbar

            return esf

    def y3_int(self, diagonal, component):
        x_pro = self.x_pro
        z = self.z
        d = self.d

        if component == 0:
            min_y_3 = bounds_dict['y3_min'][0]
            max_y_3 = bounds_dict['y3_max'][0]

            y_3 = []
            for t in diagonal:
                x_pro_0 = x_pro[0] + t * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                d_0 = d[0] + t * (d[1] - d[0])
                temp = self.y3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 0)
                y_3.append(temp)
            y_3 = np.asarray(y_3)
            y_3 = (y_3 - min_y_3) / (max_y_3 - min_y_3)

            return y_3

        if component == 1:
            min_y_3 = bounds_dict['y3_min'][1]
            max_y_3 = bounds_dict['y3_max'][1]

            y_3 = []
            for t in diagonal:
                x_pro_0 = x_pro[0] + t * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                d_0 = d[0] + t * (d[1] - d[0])
                temp = self.y3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 1)
                y_3.append(temp)
            y_3 = np.asarray(y_3)
            y_3 = (y_3 - min_y_3) / (max_y_3 - min_y_3)

            return y_3

        if component == 2:
            min_y_3 = bounds_dict['y3_min'][2]
            max_y_3 = bounds_dict['y3_max'][2]

            y_3 = []
            for t in diagonal:
                x_pro_0 = x_pro[0] + t * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                d_0 = d[0] + t * (d[1] - d[0])
                temp = self.y3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 2)
                y_3.append(temp)

            y_3 = np.asarray(y_3)
            y_3 = (y_3 - min_y_3) / (max_y_3 - min_y_3)

            return y_3

    @staticmethod
    def y31(x_pro, d, wbe, component):
        tbar = abs(x_pro) * 16168.6
        esf = (d / 3.0) / tbar

        if component == 0:
            we = 3.0 * wbe * abs(esf) ** 1.05

            return we

    def y31_int(self, diagonal, component):
        x_pro = self.x_pro
        d = self.d
        wbe = self.wbe

        if component == 0:
            min_y_31 = bounds_dict['y31_min'][0]
            max_y_31 = bounds_dict['y31_max'][0]

            y_31 = []
            for t in diagonal:
                x_pro_0 = x_pro[0] + t * (x_pro[1] - x_pro[0])
                d_0 = d[0] + t * (d[1] - d[0])
                we = self.y31(x_pro_0, d_0, wbe, 0)
                y_31.append(we)

            y_31 = np.asarray(y_31)
            y_31 = (y_31 - min_y_31) / (max_y_31 - min_y_31)

            return y_31

    @staticmethod
    def y32(x_pro, drag, component):
        tbar = abs(x_pro) * 16168.6

        if component == 0:
            esf = (drag / 3.0) / tbar

            return esf

    def y32_int(self, diagonal, component):
        x_pro = self.x_pro
        d = self.d

        if component == 0:
            min_y_32 = bounds_dict['y32_min'][0]
            max_y_32 = bounds_dict['y32_max'][0]

            y_32 = []
            for t in diagonal:
                x_pro_0 = x_pro[0] + t * (x_pro[1] - x_pro[0])
                d_0 = d[0] + t * (d[1] - d[0])
                esf = self.y32(x_pro_0, d_0, 0)
                y_32.append(esf)

            y_32 = np.asarray(y_32)
            y_32 = (y_32 - min_y_32) / (max_y_32 - min_y_32)

            return y_32

    @staticmethod
    def y34(x_pro, z, component):

        if component == 0:
            tbar = abs(x_pro) * 16168.6
            sfc = 1.1324 + 1.5344 * z[2] - 3.2956E-05 * z[1] - 1.6379E-04 * tbar - 0.31623 * z[2] ** 2 + 8.2138E-06 * \
                z[2] * z[1] - 10.496E-5 * tbar * z[2] - 8.574E-11 * z[1] ** 2 + 3.8042E-9 * tbar * z[1] + 1.06E-8 * \
                tbar ** 2

            return sfc

    def y34_int(self, diagonal, component):
        x_pro = self.x_pro
        z = self.z

        if component == 0:
            min_y_34 = bounds_dict['y34_min'][0]
            max_y_34 = bounds_dict['y34_max'][0]

            y_34 = []
            for t in diagonal:
                x_pro_0 = x_pro[0] + t * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                sfc = self.y34(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], 0)
                y_34.append(sfc)

            y_34 = np.asarray(y_34)
            y_34 = (y_34 - min_y_34) / (max_y_34 - min_y_34)

            return y_34

    @staticmethod
    def g3(x_pro, z, drag, component):

        if component == 0:
            tbar = abs(x_pro) * 16168.6
            esf = (drag / 3.0) / tbar

            return 0.5 - esf

        if component == 1:
            tbar = abs(x_pro) * 16168.6
            esf = (drag / 3.0) / tbar
            return esf - 1.5

        if component == 2:
            tbar = abs(x_pro) * 16168.6
            tu_abar_pre = 11484.0 + 10856.0 * z[2] - 0.50802 * z[1] + 3200.2 * (z[2] ** 2)
            tu_abar = tu_abar_pre - 0.29326 * z[2] * z[1] + 6.8572E-6 * z[1] ** 2
            dt = tbar / (tu_abar + 0.01) - 1.0

            return dt

        if component == 3:
            temp = polynomial_function([z[2], z[1], abs(x_pro)], [2, 4, 2], [.25] * 3, "Temp")

            return temp - 1.02

    def g3_int(self, diagonal, component):
        x_pro = self.x_pro
        z = self.z
        d = self.d

        if component == 0:

            min_g_3 = bounds_dict['g3_min'][0]
            max_g_3 = bounds_dict['g3_max'][0]

            g_3 = []
            for t in diagonal:
                x_pro_0 = x_pro[0] + t * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                d_0 = d[0] + t * (d[1] - d[0])
                esf = self.g3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 0)
                g_3.append(esf)

            g_3 = np.asarray(g_3)
            g_3 = (g_3 - min_g_3) / (max_g_3 - min_g_3)

            return g_3

        if component == 1:

            min_g_3 = bounds_dict['g3_min'][1]
            max_g_3 = bounds_dict['g3_max'][1]

            g_3 = []
            for t in diagonal:
                x_pro_0 = x_pro[0] + t * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                d_0 = d[0] + t * (d[1] - d[0])
                esf = self.g3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 1)
                g_3.append(esf)

            g_3 = np.asarray(g_3)
            g_3 = (g_3 - min_g_3) / (max_g_3 - min_g_3)

            return g_3

        if component == 2:

            min_g_3 = bounds_dict['g3_min'][2]
            max_g_3 = bounds_dict['g3_max'][2]

            g_3 = []
            for t in diagonal:
                x_pro_0 = x_pro[0] + t * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                d_0 = d[0] + t * (d[1] - d[0])
                dt = self.g3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 2)
                g_3.append(dt)
            g_3 = np.asarray(g_3)
            g_3 = (g_3 - min_g_3) / (max_g_3 - min_g_3)

            return g_3

        if component == 3:

            min_g_3 = bounds_dict['g3_min'][3]
            max_g_3 = bounds_dict['g3_max'][3]

            g_3 = []
            for t in diagonal:
                x_pro_0 = x_pro[0] + t * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t * (z[5][1] - z[5][0])
                d_0 = d[0] + t * (d[1] - d[0])
                temp = self.g3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 3)
                g_3.append(temp)
            g_3 = np.asarray(g_3)
            g_3 = (g_3 - min_g_3) / (max_g_3 - min_g_3)

            return g_3


class PerformanceInterpolation:

    def __init__(self):
        self.scale = np.array([1 * 1.2, .45 * .98])
        #  :lower and upper bounds on design and coupling variables
        self.x_str = np.array([[0.4, 0.6], [.75, 1.25]])  # : lambda_max raises theta initial
        self.z = np.array([[0.034, 0.14], [20000, 45000], [.5, 1.2], [3.0, 8], [30, 190], [600, 2100]]) * self.scale
        #  :z[3][0] changes convexity of WFW & Theta; z[4] alters WT; z[5]_max changes Theta smoothness
        self.load = np.array([15000, 70000]) * self.scale
        self.we = np.array([0, 70000]) * self.scale
        self.nz = 6.0
        self.wfo = 2000.0
        self.wo = 25000.0

        self.x_aer = np.array([0.65, 1.25]) * self.scale
        self.wt = np.array([15000., 70000.]) * self.scale
        self.esf = np.array([0.5, 1.5]) * self.scale
        self.theta = np.array([0.96, 1.04])
        self.cdmin = 0.01375

        self.x_pro = np.array([0.37, .6])
        self.d = np.array([1000., 15000.])
        self.wbe = 4360

    @staticmethod
    def range(z, wt, wf, fin, sfc):

        if z[1] <= 36089.:
            theta = 1.0 - 6.875E-6 * z[1]
        else:
            theta = 0.7519
        range = 661.0 * np.sqrt(theta) * z[2] * fin / sfc * np.log(abs(wt / (wt - wf + .1)))
        return range

    def range_int(self, diagonal):
        z = self.z
        wt = np.array([15000, 70000])
        wf = np.array([5000., 25000.])
        fin = np.array([2., 12.])
        sfc = np.array([.5, 1.5])

        min_ran = bounds_dict['ran_min']
        max_ran = bounds_dict['ran_max']

        ran = []
        for t in diagonal:
            z_0 = z[0][0] + t * (z[0][1] - z[0][0])
            z_1 = z[1][0] + t * (z[1][1] - z[1][0])
            z_2 = z[2][0] + t * (z[2][1] - z[2][0])
            z_3 = z[3][0] + t * (z[3][1] - z[3][0])
            z_4 = z[4][0] + t * (z[4][1] - z[4][0])
            z_5 = z[5][0] + t * (z[5][1] - z[5][0])
            wt_0 = wt[0] + t * (wt[1] - wt[0])
            wf_0 = wf[0] + t * (wf[1] - wf[0])
            fin_0 = fin[0] + t * (fin[1] - fin[0])
            sfc_0 = sfc[0] + t * (sfc[1] - sfc[0])
            ran.append(self.range([z_0, z_1, z_2, z_3, z_4, z_5], wt_0, wf_0, fin_0, sfc_0))

        ran = np.asarray(ran)
        ran = (ran - min_ran) / (max_ran - min_ran)

        return ran


# # :range calculation
# fig = plt.figure(1)
# p = PerformanceInterpolation()
# x = np.linspace(0, 1, 100)
# y = []
# for k in range(len(x)):
#     y.append(getattr(p, 'range_int')([x[k]]))
# plt.plot(x, y)
# plt.show()


# #---------------------------------------------------------------------------------------------------------------- #
# #                       The following section generates the interpolated graphs for the output                   #
# #---------------------------------------------------------------------------------------------------------------- #

# add the output labels
# label_structure, label_aerodynamics, label_propulsion = ['y1_int', 'y11_int', 'y12_int', 'y14_int', 'g1_int'],\
#                                                         ['y2_int', 'y21_int', 'y23_int', 'y24_int', 'g2_int'],\
#                                                         ['y3_int', 'y31_int', 'y32_int', 'y34_int', 'g3_int']
# label = [label_structure, label_aerodynamics, label_propulsion]
#
# # :Make a list of the discipline classes
# classes = np.array([StructureInterpolation, AerodynamicsInterpolation, PropulsionInterpolation])
# # :Make a lst of the number of outputs of each coupling in same order as labels
# number_outputs = [[3, 1, 2, 2, 7], [3, 1, 1, 1, 1], [2, 1, 1, 1, 4]]
# # : Add the abscissa for interpolation
# x = np.linspace(0, 1, 25)
# # add the counter for subplots
# counter = [1 + 3 * k for k in range(5)] + [2 + 3 * k for k in range(5)] + [3 + 3 * k for k in range(5)]
# fig = plt.figure(2)
# for instance in classes:  # :iterate through classes
#     class_index = list(classes).index(instance)
#     class_instance = classes[class_index]()
#     for output_def in label[class_index]:  # :iterate through output definition methods
#         output_index = list(label[class_index]).index(output_def)
#         ax = fig.add_subplot(5, 3, counter.pop(0))
#         ax.set_xlabel(output_def, fontsize=10)
#         component = number_outputs[class_index][output_index]
#         for component in range(component):  # :iterate through number of components
#             y = []
#             for k in range(len(x)):
#                 y.append(getattr(class_instance, output_def)([x[k]], component))
#             plt.plot(x, y)
#             plt.yticks(np.arange(0, 2, 1))
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.show()
#
# # # ---------------------------------------------------------------------------------------------------------------- #
# # #        The following section generates the representative cuts for the original SSBJ problem                     #
# # # ---------------------------------------------------------------------------------------------------------------- #
#
# fig = plt.figure(3)
# number_outputs = [[3, 1, 2, 2, 7], [3, 1, 1, 1, 1], [2, 1, 1, 1, 4]]
# label_structure, label_aerodynamics, label_propulsion = ['y1_int', 'y11_int', 'y12_int', 'y14_int', 'g1_int'],\
#                                                         ['y2_int', 'y21_int', 'y23_int', 'y24_int', 'g2_int'],\
#                                                         ['y3_int', 'y31_int', 'y32_int', 'y34_int', 'g3_int']
# label = [label_structure, label_aerodynamics, label_propulsion]
# classes = np.array([StructureInterpolation, AerodynamicsInterpolation, PropulsionInterpolation])
# x = np.linspace(0, 1, 25)
# for i in range(36):
#     ax = fig.add_subplot(6, 6, i + 1)
#     random_class = [random.choice(range(3)), random.choice(range(3))]
#     class_output = [classes[random_class[0]](), classes[random_class[1]]()]
#     random_output = [random.choice(range(5)), random.choice(range(5))]
#     output = [label[random_class[0]][random_output[0]], label[random_class[1]][random_output[1]]]
#     component = [random.choice(range(number_outputs[random_class[0]][random_output[0]])),
#                  random.choice(range(number_outputs[random_class[1]][random_output[1]]))]
#     y1 = [[], []]
#     for k in range(len(x)):
#         y1[0].append(getattr(class_output[0], output[0])([x[k]], component[0]))
#         y1[1].append(getattr(class_output[1], output[1])([x[k]], component[1]))
#     plt.plot(y1[0], y1[1])
#     plt.yticks(np.arange(0, 3, 1))
#     ax.set_xlabel(output[0], fontsize=10)
#     ax.set_ylabel(output[1], fontsize=10)
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.show()

# # : make representative cut of y2 (3 components) vs y2 (1 component)
# # TODO: remove min max interpolations from y12 and y2
# fig = plt.figure(4)
# x = np.linspace(0, 1, 10)
# s = StructureInterpolation()
# a = AerodynamicsInterpolation()
# p = PropulsionInterpolation()
#
# y12 = []
# y2_0 = []
# y2_1 = []
# y2_2 = []
# for k in range(len(x)):
#     y12.append(getattr(s, 'y12_int')([x[k]], 0))
#     y2_0.append(getattr(a, 'y2_int')([x[k]], 0))
#     y2_1.append(getattr(a, 'y2_int')([x[k]], 1))
#     y2_2.append(getattr(a, 'y2_int')([x[k]], 2))
# plt.plot(y12, y2_0)
# plt.plot(y12, y2_1)
# plt.plot(y12, y2_2)
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.show()