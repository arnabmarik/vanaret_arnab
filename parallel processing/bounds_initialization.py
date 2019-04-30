
# Imports
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pylab as plt
from ssbjkadmos.utils.math import polynomial_function
import random
import pickle


class StructureBounds:

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

    def y12_int(self, component):

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
            y_12 = []
            for a1 in range(100):
                t1 = np.linspace(0, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                we_0 = we[0] + t1 * (we[1] - we[0])
                wt = self.y12([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 0)
                y_12.append(wt)
            min_y_12 = min(y_12)
            max_y_12 = max(y_12)

            return min_y_12, max_y_12

        if component == 1:
            y_12 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                we_0 = we[0] + t1 * (we[1] - we[0])
                theta = self.y12([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 1)
                y_12.append(theta)
            min_y_12 = min(y_12)
            max_y_12 = max(y_12)

            return min_y_12, max_y_12

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

    def y14_int(self, component):

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
            y_14 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                we_0 = we[0] + t1 * (we[1] - we[0])
                wt = self.y14([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 0)
                y_14.append(wt)
            min_y_14 = min(y_14)
            max_y_14 = max(y_14)

            return min_y_14, max_y_14

        if component == 1:
            y_14 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                we_0 = we[0] + t1 * (we[1] - we[0])
                wf = self.y14([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 1)
                y_14.append(wf)
            min_y_14 = min(y_14)
            max_y_14 = max(y_14)

            return min_y_14, max_y_14

    @staticmethod
    def y11(z, w_fo, component):

        # common calculations
        t = z[0] * z[5] / (np.sqrt(abs(z[5] * z[3])))

        # calculations for specific components
        if component == 0:
            wfw = 5.0 / 18.0 * abs(z[5]) * 2.0 / 3.0 * t * 42.5
            wf = wfw + w_fo

            return wf

    def y11_int(self, component):

        z = self.z
        wfo = 2000.0

        # calculation of interpolated values for each component
        if component == 0:
            y_11 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                wt = self.y11([z_0, z_1, z_2, z_3, z_4, z_5], wfo, 0)
                y_11.append(wt)
            min_y_11 = min(y_11)
            max_y_11 = max(y_11)

            return min_y_11, max_y_11

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
                        / abs(np.cos(z[4] * np.pi / 180.)))
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

    def y1_int(self, component):

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
            y_1 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                we_0 = we[0] + t1 * (we[1] - we[0])
                wt = self.y1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 0)
                y_1.append(wt)
            min_y_1 = min(y_1)
            max_y_1 = max(y_1)

            return min_y_1, max_y_1

        if component == 1:
            y_1 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                we_0 = we[0] + t1 * (we[1] - we[0])
                wfw = self.y1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 1)
                y_1.append(wfw)
            min_y_1 = min(y_1)
            max_y_1 = max(y_1)

            return min_y_1, max_y_1

        if component == 2:
            y_1 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                we_0 = we[0] + t1 * (we[1] - we[0])
                wf = self.y1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, we_0, nz, wfo, wo, 2)
                y_1.append(wf)
            min_y_1 = min(y_1)
            max_y_1 = max(y_1)

            return min_y_1, max_y_1

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

    def g1_int(self, component):

        x_str = self.x_str
        z = self.z
        load = self.load

        if component == 0:
            g_1 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                sigma1 = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 0)
                g_1.append(sigma1)
            min_g_1_1 = min(g_1)
            max_g_1_1 = max(g_1)

            return min_g_1_1, max_g_1_1

        if component == 1:
            g_1 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                sigma2 = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 1)
                g_1.append(sigma2)
            min_g_1_2 = min(g_1)
            max_g_1_2 = max(g_1)

            return min_g_1_2, max_g_1_2

        if component == 2:
            g_1 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                sigma3 = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 2)
                g_1.append(sigma3)
            min_g_1_3 = min(g_1)
            max_g_1_3 = max(g_1)

            return min_g_1_3, max_g_1_3

        if component == 3:
            g_1 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                sigma4 = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 3)
                g_1.append(sigma4)
            min_g_1_4 = min(g_1)
            max_g_1_4 = max(g_1)

            return min_g_1_4, max_g_1_4

        if component == 4:
            g_1 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                sigma5 = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 4)
                g_1.append(sigma5)
            min_g_1_5 = min(g_1)
            max_g_1_5 = max(g_1)

            return min_g_1_5, max_g_1_5

        if component == 5:
            g_1 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                theta = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 5)
                g_1.append(theta)
            min_g_1_6 = min(g_1)
            max_g_1_6 = max(g_1)

            return min_g_1_6, max_g_1_6

        if component == 6:
            g_1 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_str_0 = x_str[0][0] + t1 * (x_str[0][1] - x_str[0][0])
                x_str_1 = x_str[1][0] + t1 * (x_str[1][1] - x_str[1][0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                l_0 = load[0] + t1 * (load[1] - load[0])
                theta = self.g1([x_str_0, x_str_1], [z_0, z_1, z_2, z_3, z_4, z_5], l_0, 6)
                g_1.append(theta)
            min_g_1_7 = min(g_1)
            max_g_1_7 = max(g_1)

            return min_g_1_7, max_g_1_7


class AerodynamicsBounds:

    def __init__(self):
        self.z = StructureBounds().z
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

    def y21_int(self, component):
        wt = self.wt

        if component == 0:
            y_21 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                wt_0 = wt[0] + t1 * (wt[1] - wt[0])
                load = self.y21(wt_0, 0)
                y_21.append(load)
            min_y_21 = min(y_21)
            max_y_21 = max(y_21)

            return min_y_21, max_y_21

    @staticmethod
    def y23(x_aer, z, w_t, esf, theta, cdmin, component):

        if component == 0:
            if z[1] <= 36089.0:
                v = 1116.39 * z[2] * np.sqrt(abs(1.0 - 6.875E-6 * z[1]))
                rho = 2.377E-3 * (1. - 6.875E-6 * z[1]) ** 4.2561
            else:
                v = 968.1 * abs(z[2])
                rho = 2.377E-3 * 0.2971 * np.exp((36089.0 - z[1]) / 20806.7)
            cl = w_t / (0.5 * rho * (v ** 2) * z[5])
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

    def y23_int(self, component):
        x_aer = self.x_aer
        z = self.z
        wt = self.wt
        esf = self.esf
        theta = self.theta
        cdmin = self.cdmin

        if component == 0:
            y_23 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
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
                d = self.y23(x_aer_0, [z_0, z_1, z_2, z_3, z_4, z_5], wt_0, esf_0, theta_0, cdmin, 0)
                y_23.append(d)

            min_y_23 = min(y_23)
            max_y_23 = max(y_23)

            return min_y_23, max_y_23

    @staticmethod
    def y24(x_aer, z, w_t, esf, theta, cdmin, component):

        if component == 0:
            if z[1] <= 36089.0:
                v = 1116.39 * z[2] * np.sqrt(abs(1.0 - 6.875E-6 * z[1]))
                rho = 2.377E-3 * (1. - 6.875E-6 * z[1]) ** 4.2561
            else:
                v = 968.1 * abs(z[2])
                rho = 2.377E-3 * 0.2971 * np.exp((36089.0 - z[1]) / 20806.7)
            cl = w_t / (0.5 * rho * (v ** 2) * z[5])
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
            fin = w_t / d

            return fin

    def y24_int(self, component):
        x_aer = self.x_aer
        z = self.z
        wt = self.wt
        esf = self.esf
        theta = self.theta
        cdmin = self.cdmin

        if component == 0:
            y_24 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
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
                fin = self.y24(x_aer_0, [z_0, z_1, z_2, z_3, z_4, z_5], wt_0, esf_0, theta_0, cdmin, 0)
                y_24.append(fin)

            min_y_24 = min(y_24)
            max_y_24 = max(y_24)

            return min_y_24, max_y_24

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
            cl = w_t / (0.5 * rho * (v ** 2) * z[5])
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
            cl = w_t / (0.5 * rho * (v ** 2) * z[5])
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
            fin = w_t / d

            return fin

    def y2_int(self, component):

        x_aer = self.x_aer
        z = self.z
        wt = self.wt
        esf = self.esf
        theta = self.theta
        cdmin = self.cdmin

        if component == 0:
            y_2 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
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

            return min_y_2, max_y_2

        if component == 1:
            y_2 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
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

            return min_y_2, max_y_2

        if component == 2:
            y_2 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
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

            return min_y_2, max_y_2

    @staticmethod
    def g2(z, component):

        if component == 0:
            dpdx = polynomial_function([z[0]], [1], [.25], "dpdx")

            return dpdx - 1.04

    def g2_int(self, component):
        z = self.z

        if component == 0:
            g_2 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
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

            return min_g_2, max_g_2


class PropulsionBounds:

    def __init__(self):
        self.z = StructureBounds().z
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

    def y3_int(self, component):
        x_pro = self.x_pro
        z = self.z
        d = self.d

        if component == 0:
            y_3 = []
            for a in range(100):
                t1 = np.linspace(0.01, 1, 100)[a]
                x_pro_0 = x_pro[0] + t1 * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                d_0 = d[0] + t1 * (d[1] - d[0])
                temp = self.y3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 0)
                y_3.append(temp)

            min_y_3 = min(y_3)
            max_y_3 = max(y_3)

            return min_y_3, max_y_3

        if component == 1:
            y_3 = []
            for a in range(100):
                t1 = np.linspace(0.01, 1, 100)[a]
                x_pro_0 = x_pro[0] + t1 * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                d_0 = d[0] + t1 * (d[1] - d[0])
                esf = self.y3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 1)
                y_3.append(esf)

            min_y_3 = min(y_3)
            max_y_3 = max(y_3)

            return min_y_3, max_y_3

        if component == 2:
            y_3 = []
            for a in range(100):
                t1 = np.linspace(0.01, 1, 100)[a]
                x_pro_0 = x_pro[0] + t1 * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                d_0 = d[0] + t1 * (d[1] - d[0])
                esf = self.y3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 2)
                y_3.append(esf)

            min_y_3 = min(y_3)
            max_y_3 = max(y_3)

            return min_y_3, max_y_3

    @staticmethod
    def y31(x_pro, d, wbe, component):
        tbar = abs(x_pro) * 16168.6
        esf = (d / 3.0) / tbar

        if component == 0:
            we = 3.0 * wbe * abs(esf) ** 1.05

            return we

    def y31_int(self, component):
        x_pro = self.x_pro
        d = self.d
        wbe = self.wbe

        if component == 0:
            y_31 = []
            for a in range(100):
                t1 = np.linspace(0.01, 1, 100)[a]
                x_pro_0 = x_pro[0] + t1 * (x_pro[1] - x_pro[0])
                d_0 = d[0] + t1 * (d[1] - d[0])
                we = self.y31(x_pro_0, d_0, wbe, 0)
                y_31.append(we)

            min_y_31 = min(y_31)
            max_y_31 = max(y_31)

            return min_y_31, max_y_31

    @staticmethod
    def y32(x_pro, drag, component):
        tbar = abs(x_pro) * 16168.6

        if component == 0:
            esf = (drag / 3.0) / tbar

            return esf

    def y32_int(self, component):
        x_pro = self.x_pro
        d = self.d

        if component == 0:
            y_32 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_pro_0 = x_pro[0] + t1 * (x_pro[1] - x_pro[0])
                d_0 = d[0] + t1 * (d[1] - d[0])
                esf = self.y32(x_pro_0, d_0, 0)
                y_32.append(esf)

            min_y_32 = min(y_32)
            max_y_32 = max(y_32)

            return min_y_32, max_y_32

    @staticmethod
    def y34(x_pro, z, component):

        if component == 0:
            tbar = abs(x_pro) * 16168.6
            sfc = 1.1324 + 1.5344 * z[2] - 3.2956E-05 * z[1] - 1.6379E-04 * tbar - 0.31623 * z[2] ** 2 + 8.2138E-06 * \
                z[2] * z[1] - 10.496E-5 * tbar * z[2] - 8.574E-11 * z[1] ** 2 + 3.8042E-9 * tbar * z[1] + 1.06E-8 * \
                tbar ** 2

            return sfc

    def y34_int(self,  component):
        x_pro = self.x_pro
        z = self.z

        if component == 0:
            y_34 = []
            for a1 in range(100):
                t1 = np.linspace(0.01, 1, 100)[a1]
                x_pro_0 = x_pro[0] + t1 * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                sfc = self.y34(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], 0)
                y_34.append(sfc)

            min_y_34 = min(y_34)
            max_y_34 = max(y_34)

            return min_y_34, max_y_34

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
            dt = tbar / tu_abar - 1.0

            return dt

        if component == 3:
            temp = polynomial_function([z[2], z[1], abs(x_pro)], [2, 4, 2], [.25] * 3, "Temp")

            return temp - 1.02

    def g3_int(self, component):
        x_pro = self.x_pro
        z = self.z
        d = self.d

        if component == 0:
            g_3 = []
            for a in range(100):
                t1 = np.linspace(0.01, 1, 100)[a]
                x_pro_0 = x_pro[0] + t1 * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                d_0 = d[0] + t1 * (d[1] - d[0])
                esf = self.g3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 0)
                g_3.append(esf)

            min_g_3 = min(g_3)
            max_g_3 = max(g_3)

            return min_g_3, max_g_3

        if component == 1:

            g_3 = []
            for a in range(100):
                t1 = np.linspace(0.01, 1, 100)[a]
                x_pro_0 = x_pro[0] + t1 * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                d_0 = d[0] + t1 * (d[1] - d[0])
                esf = self.g3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 1)
                g_3.append(esf)

            min_g_3 = min(g_3)
            max_g_3 = max(g_3)

            return min_g_3, max_g_3

        if component == 2:

            g_3 = []
            for a in range(100):
                t1 = np.linspace(0.01, 1, 100)[a]
                x_pro_0 = x_pro[0] + t1 * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                d_0 = d[0] + t1 * (d[1] - d[0])
                dt = self.g3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 2)
                g_3.append(dt)

            min_g_3 = min(g_3)
            max_g_3 = max(g_3)

            return min_g_3, max_g_3

        if component == 3:

            g_3 = []
            for a in range(100):
                t1 = np.linspace(0.01, 1, 100)[a]
                x_pro_0 = x_pro[0] + t1 * (x_pro[1] - x_pro[0])
                z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
                z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
                z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
                z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
                z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
                z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
                d_0 = d[0] + t1 * (d[1] - d[0])
                temp = self.g3(x_pro_0, [z_0, z_1, z_2, z_3, z_4, z_5], d_0, 3)
                g_3.append(temp)

            min_g_3 = min(g_3)
            max_g_3 = max(g_3)

            return min_g_3, max_g_3


class PerformanceBounds:

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
        range = 661.0 * np.sqrt(theta) * z[2] * fin / sfc * np.log(abs(wt / (wt - wf)))
        return range

    def range_int(self):
        z = self.z
        wt = np.array([15000, 70000])
        wf = np.array([5000., 25000.])
        fin = np.array([2., 12.])
        sfc = np.array([.5, 1.5])

        ran = []
        for a in range(100):
            t1 = np.linspace(0.01, 1, 100)[a]
            z_0 = z[0][0] + t1 * (z[0][1] - z[0][0])
            z_1 = z[1][0] + t1 * (z[1][1] - z[1][0])
            z_2 = z[2][0] + t1 * (z[2][1] - z[2][0])
            z_3 = z[3][0] + t1 * (z[3][1] - z[3][0])
            z_4 = z[4][0] + t1 * (z[4][1] - z[4][0])
            z_5 = z[5][0] + t1 * (z[5][1] - z[5][0])
            wt_0 = wt[0] + t1 * (wt[1] - wt[0])
            wf_0 = wf[0] + t1 * (wf[1] - wf[0])
            fin_0 = fin[0] + t1 * (fin[1] - fin[0])
            sfc_0 = sfc[0] + t1 * (sfc[1] - sfc[0])
            ran.append(self.range([z_0, z_1, z_2, z_3, z_4, z_5], wt_0, wf_0, fin_0, sfc_0))

        min_ran = min(ran)
        max_ran = max(ran)

        return min_ran, max_ran


def create_bounds_dict():
    # initialize a dictionary to store the lower and upper bounds on the original components
    bounds_dict = dict()

    sb = StructureBounds()
    ab = AerodynamicsBounds()
    pb = PropulsionBounds()
    perb = PerformanceBounds()

    # create bounds on every original output variable
    bounds_dict['y12_min'] = [sb.y12_int(0)[0], sb.y12_int(1)[0]]
    bounds_dict['y12_max'] = [sb.y12_int(0)[1], sb.y12_int(1)[1]]
    bounds_dict['y14_min'] = [sb.y14_int(0)[0], sb.y14_int(1)[0]]
    bounds_dict['y14_max'] = [sb.y14_int(0)[1], sb.y14_int(1)[1]]
    bounds_dict['y11_min'] = [sb.y11_int(0)[0]]
    bounds_dict['y11_max'] = [sb.y11_int(0)[1]]
    bounds_dict['y1_min'] = [sb.y1_int(0)[0], sb.y1_int(1)[0], sb.y1_int(2)[0]]
    bounds_dict['y1_max'] = [sb.y1_int(0)[1], sb.y1_int(1)[1], sb.y1_int(2)[1]]
    bounds_dict['g1_min'] = [sb.g1_int(0)[0], sb.g1_int(1)[0], sb.g1_int(2)[0], sb.g1_int(3)[0], sb.g1_int(4)[0],
                             sb.g1_int(5)[0], sb.g1_int(6)[0]]
    bounds_dict['g1_max'] = [sb.g1_int(0)[1], sb.g1_int(1)[1], sb.g1_int(2)[1], sb.g1_int(3)[1], sb.g1_int(4)[1],
                             sb.g1_int(5)[1], sb.g1_int(6)[1]]
    bounds_dict['y21_min'] = [ab.y21_int(0)[0]]
    bounds_dict['y21_max'] = [ab.y21_int(0)[1]]
    bounds_dict['y23_min'] = [ab.y23_int(0)[0]]
    bounds_dict['y23_max'] = [ab.y23_int(0)[1]]
    bounds_dict['y24_min'] = [ab.y24_int(0)[0]]
    bounds_dict['y24_max'] = [ab.y24_int(0)[1]]
    bounds_dict['y2_min'] = [ab.y2_int(0)[0], ab.y2_int(1)[0], ab.y2_int(2)[0]]
    bounds_dict['y2_max'] = [ab.y2_int(0)[1], ab.y2_int(1)[1], ab.y2_int(2)[1]]
    bounds_dict['g2_min'] = [ab.g2_int(0)[0]]
    bounds_dict['g2_max'] = [ab.g2_int(0)[1]]
    bounds_dict['y3_min'] = [pb.y3_int(0)[0], pb.y3_int(1)[0], pb.y3_int(2)[0]]
    bounds_dict['y3_max'] = [pb.y3_int(0)[1], pb.y3_int(1)[1], pb.y3_int(2)[1]]
    bounds_dict['y31_min'] = [pb.y31_int(0)[0]]
    bounds_dict['y31_max'] = [pb.y31_int(0)[1]]
    bounds_dict['y32_min'] = [pb.y32_int(0)[0]]
    bounds_dict['y32_max'] = [pb.y32_int(0)[1]]
    bounds_dict['y34_min'] = [pb.y34_int(0)[0]]
    bounds_dict['y34_max'] = [pb.y34_int(0)[1]]
    bounds_dict['g3_min'] = [pb.g3_int(0)[0], pb.g3_int(1)[0], pb.g3_int(2)[0], pb.g3_int(3)[0]]
    bounds_dict['g3_max'] = [pb.g3_int(0)[1], pb.g3_int(1)[1], pb.g3_int(2)[1], pb.g3_int(3)[1]]
    bounds_dict['ran_min'] = perb.range_int()[0]
    bounds_dict['ran_max'] = perb.range_int()[1]

    pickle.dump(bounds_dict, open("bounds_dict.p", "wb"))
