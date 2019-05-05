"""
This file contains the individual couplings and constraint outputs
 for the SSBJ problem as per the Vanaret paper, wherein each output is treated as a discipline
 which takes as input a vector of Nx dimensions and outputs a vector of size ny
 """

# Imports
import numpy as np
import pickle
import os
from interpolated_functions import StructureInterpolation
from interpolated_functions import AerodynamicsInterpolation
from interpolated_functions import PropulsionInterpolation
from interpolated_functions import PerformanceInterpolation

np.set_printoptions(threshold=np.inf)


class Structure:

    def __init__(self):

        self.str_int = StructureInterpolation()  # This class gives component wise interpolation for structural outputs
        with open("dependency_matrix.p", "rb") as f:
            self.dependency_matrix = pickle.load(f)
        with open("component_dependency.p", "rb") as f:
            self.component_dependency = pickle.load(f)

        # load parameters related to translating the scaled constraints
        with open("constraint_params.p", "rb") as f:
            a = pickle.load(f)
            [self.p_g1, self.p_g2, self.p_g3] = a[0]
            [self.alpha_g1, self.alpha_g2, self.alpha_g3] = a[1]
            [self.mu_g1, self.mu_g2, self.mu_g3] = a[2]
        self.p = self.p_g1  # percentage of constraints allowed to be active initially
        with open("str_count.p", "rb") as f:
            str_count = pickle.load(f)
            str_count += 1
        with open("str_count.p", "wb") as f:
            pickle.dump(str_count, f)

    def y1(self, nx, ny, x_des):
        """This output corresponds to total weight(WT) and partial fuel weight(wfw)"""

        c_d, a1, output = self.component_dependency['y_1'], self.dependency_matrix, []
        for i in range(ny):
            sum_i, row = [], a1[i]
            sum_i.append(np.sum(row))
            assign, y = c_d[i], []
            [y.append(self.str_int.y1_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)
        return output

    def y11(self, nx, ny, x_des):
        """This output corresponds to total weight(WT) and temperature ratio(theta)"""

        [c_d, a1, output] = [self.component_dependency['y_11'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[1 * ny + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # :x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.str_int.y11_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def y12(self, nx, ny, x_des):
        """This output corresponds to total weight(WT) and temperature ratio(theta)"""

        [c_d, a1, output] = [self.component_dependency['y_12'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[2 * ny + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # :x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.str_int.y12_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def y14(self, nx, ny, x_des):
        """This output corresponds to total weight(WT) and fuel weight(WF)"""

        [c_d, a1, output] = [self.component_dependency['y_14'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[3 * ny + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # :x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.str_int.y14_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def g1_unscaled(self, nx, ny, x_des):
        """This output corresponds to structural and temperature ratio constraints"""

        [c_d, a1, output] = [self.component_dependency['g_1'], self.dependency_matrix, []]
        for i in range(nx):
            [sum_i, row] = [[], a1[4 * ny + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # :x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.str_int.g1_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def g1(self, nx, ny, x_des):
        """scaling the constraints from violated to active/inactive status"""

        # :evaluating unscaled constraints at the initial point
        g_1_0 = self.g1_unscaled(nx, ny, .5 * np.ones(4 * nx + 5 * ny))

        # :evaluating unscaled constraints:
        g_1 = self.g1_unscaled(nx, ny, x_des)

        # :define the threshold "tow" to translate the scaled constraint
        # :alpha determines to what extent the inactive constraints are satisfied
        tow, alpha = [], self.alpha_g1
        [tow.append(i) if self.mu_g1[list(g_1_0).index(i)] < self.p else tow.append(alpha + (1 - alpha) * i) for i in g_1_0]

        # define the translated constraint
        g_1_translated = []
        [g_1_translated.append(g_1[i] - tow[i]) for i in range(nx)]
        return g_1_translated


class Aerodynamics:

    def __init__(self):
        self.aer_int = AerodynamicsInterpolation()  # This class gives component wise interpolation for structural outputs
        with open("dependency_matrix.p", "rb") as f:
            self.dependency_matrix = pickle.load(f)
        with open("component_dependency.p", "rb") as f:
            self.component_dependency = pickle.load(f)

        # load parameters related to translating the scaled constraints
        with open("constraint_params.p", "rb") as f:
            a = pickle.load(f)
            [self.p_g1, self.p_g2, self.p_g3] = a[0]
            [self.alpha_g1, self.alpha_g2, self.alpha_g3] = a[1]
            [self.mu_g1, self.mu_g2, self.mu_g3] = a[2]
        self.p = self.p_g2  # percentage of constraints allowed to be active initially
        with open("aer_count.p", "rb") as f:
            aer_count = pickle.load(f)
            aer_count += 1
        with open("aer_count.p", "wb") as f:
            pickle.dump(aer_count, f)

    def y2(self, nx, ny, x_des):
        """This output corresponds to lift, drag and lift-to-drag constraints"""

        c_d, a1, output = self.component_dependency['y_2'], self.dependency_matrix, []
        for i in range(ny):
            [sum_i, row] = [[], a1[4 * ny + nx + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.aer_int.y2_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def y21(self, nx, ny, x_des):
        """This output corresponds to load"""

        [c_d, a1, output] = [self.component_dependency['y_21'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[5 * ny + nx + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.aer_int.y21_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def y23(self, nx, ny, x_des):
        """This output corresponds to drag"""

        [c_d, a1, output] = [self.component_dependency['y_23'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[6 * ny + nx + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.aer_int.y23_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def y24(self, nx, ny, x_des):
        """This output corresponds to lift to drag ratio"""

        [c_d, a1, output] = [self.component_dependency['y_24'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[7 * ny + nx + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.aer_int.y24_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def g2_unscaled(self, nx, ny, x_des):
        """This output corresponds to pressure gradient constraints"""

        [c_d, a1, output] = [self.component_dependency['g_2'], self.dependency_matrix, []]
        for i in range(nx):
            [sum_i, row] = [[], a1[8 * ny + nx + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.aer_int.g2_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def g2(self, nx, ny, x_des):
        """scaling the constraints from violated to active/inactive status"""

        # :evaluating constraints at the initial point
        g_2_0 = self.g2_unscaled(nx, ny, .5 * np.ones(4 * nx + 5 * ny))

        # :evaluating unscaled constraints:
        g_2 = self.g2_unscaled(nx, ny, x_des)

        tow, alpha = [], self.alpha_g2
        [tow.append(i) if self.mu_g2[list(g_2_0).index(i)] < self.p else tow.append(alpha + (1 - alpha) * i) for i in g_2_0]

        # define the translated constraint
        g_2_translated = []
        [g_2_translated.append(g_2[i] - tow[i]) for i in range(nx)]
        return g_2_translated


class Propulsion:

    def __init__(self):
        # This class generates  the dependency matrix and componenent dependency mapping.
        self.pro_int = PropulsionInterpolation()  # This class gives component wise interpolation for structural outputs
        with open("dependency_matrix.p", "rb") as f:
            self.dependency_matrix = pickle.load(f)
        with open("component_dependency.p", "rb") as f:
            self.component_dependency = pickle.load(f)

        # load parameters related to translating the scaled constraints
        with open("constraint_params.p", "rb") as f:
            a = pickle.load(f)
            [self.p_g1, self.p_g2, self.p_g3] = a[0]
            [self.alpha_g1, self.alpha_g2, self.alpha_g3] = a[1]
            [self.mu_g1, self.mu_g2, self.mu_g3] = a[2]
        self.p = self.p_g3  # percentage of constraints allowed to be active initially
        with open("pro_count.p", "rb") as f:
            pro_count = pickle.load(f)
            pro_count += 1
        with open("pro_count.p", "wb") as f:
            pickle.dump(pro_count, f)

    def y3(self, nx, ny, x_des):
        """This output corresponds to temperature and engine scale factor"""

        [c_d, a1, output] = [self.component_dependency['y_3'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[8 * ny + 2 * nx + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.pro_int.y3_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)
        return output

    def y31(self, nx, ny, x_des):
        """This output corresponds to engine weight"""

        [c_d, a1, output] = [self.component_dependency['y_31'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[9 * ny + 2 * nx + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.pro_int.y31_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def y32(self, nx, ny, x_des):
        """This output corresponds to engine scale factor"""

        [c_d, a1, output] = [self.component_dependency['y_32'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[10 * ny + 2 * nx + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.pro_int.y32_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def y34(self, nx, ny, x_des):
        """This output corresponds to specific fuel consumption"""

        [c_d, a1, output] = [self.component_dependency['y_34'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[11 * ny + 2 * nx + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.pro_int.y34_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def g3_unscaled(self, nx, ny, x_des):
        [c_d, a1, output] = [self.component_dependency['g_3'], self.dependency_matrix, []]
        for i in range(nx):
            [sum_i, row] = [[], a1[12 * ny + 2 * nx + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.pro_int.g3_int([x_des[k]], assign - 1)) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def g3(self, nx, ny, x_des):
        """scaling the constraints from violated to active/inactive status"""

        # :evaluating constraints at the initial point
        g_3_0 = self.g3_unscaled(nx, ny, .5 * np.ones(4 * nx + 5 * ny))

        # :evaluating unscaled constraints:
        g_3 = self.g3_unscaled(nx, ny, x_des)

        # :define the threshold "tow" to translate the scaled constraint
        # :alpha determines to what extent the inactive constraints are satisfied
        tow, alpha = [], self.alpha_g3
        [tow.append(i) if self.mu_g3[list(g_3_0).index(i)] < self.p else tow.append(alpha + (1 - alpha) * i) for i in g_3_0]

        # define the translated constraint
        g_3_translated = []
        [g_3_translated.append(g_3[i] - tow[i]) for i in range(nx)]
        return g_3_translated


class Performance:

    def __init__(self):
        # This class generates  the dependency matrix and component dependency mapping.
        self.per_int = PerformanceInterpolation()  # This class gives component wise interpolation for structural outputs
        with open("dependency_matrix.p", "rb") as f:
            self.dependency_matrix = pickle.load(f)
        with open("component_dependency.p", "rb") as f:
            self.component_dependency = pickle.load(f)

    def range14(self, nx, ny, x_des):
        """This output corresponds to total weight(WT) and fuel weight(WF)"""
        [c_d, a1, output] = [self.component_dependency['y_14'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[3 * ny + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # :x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.per_int.range_int([x_des[k]])) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def range24(self, nx, ny, x_des):
        """This output corresponds to total weight(WT) and fuel weight(WF)"""
        [c_d, a1, output] = [self.component_dependency['y_24'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[7 * ny + nx + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # :x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.per_int.range_int([x_des[k]])) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def range34(self, nx, ny, x_des):
        """This output corresponds to total weight(WT) and fuel weight(WF)"""
        [c_d, a1, output] = [self.component_dependency['y_34'], self.dependency_matrix, []]
        for i in range(ny):
            [sum_i, row] = [[], a1[11 * ny + 2 * nx + i]]
            sum_i.append(np.sum(row))
            [assign, y] = [c_d[i], []]
            # :x_des = np.random.random_sample(4 * nx + 5 * ny)  # this is an instance of the design vector
            [y.append(self.per_int.range_int([x_des[k]])) for k in range(4 * nx + 5 * ny) if row[k] == 1]
            output.append(np.sum(y) * 1 / sum_i)

        return output

    def range(self, nx, ny, x_des):
        range14 = self.range14(nx, ny, x_des)
        range24 = self.range24(nx, ny, x_des)
        range34 = self.range34(nx, ny, x_des)

        return -1 * (np.sum(range14) + np.sum(range24) + np.sum(range34))


# #
# # s1 = Structure()
# # # s2 = Aerodynamics()
# s3 = Propulsion()
# # # # s4 = Performance()
# nx, ny = 3, 4
# # # # c = np.ones(nx) * 1
# a = s3.y3(nx, ny, .4 * np.ones(4 * nx + 5 * ny))
# print(a)

