"""
This file contains the vanaret class which is
used to generate the dependency matrix between inputs and outputs of the SSBJ problem,
as shown in the paper: Charlie Vanaret, Francois Gallard, and Joaquim Martins.
"On the Consequences of the "No Free Lunch" Theorem for Optimization on the Choice of an Appropriate MDO Architecture"
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag as bd
import pickle

# :np.set_printoptions(threshold=np.inf)


class LargeRandomMatrix:
    """The class generated the large random depedency matrix"""

    def __init__(self, nx_large, ny_large, d_large):
        self.nx_str = self.nx_str_bias = self.nx_aero = self.nx_prop = self.nx_shared = nx_large
        self.ny_str = self.ny_str_bias = self.ny_aero = self.ny_prop = self.ny_shared = ny_large
        self.d_str = self.d_str_bias = self.d_aero = self.d_prop = self.d_shared = d_large

    def submatrix_structural(self):
        """Define the structural part of the component dependency matrix"""

        # :allocate the input variables and couplings
        x = []
        [x.append(i1) for i1 in range(self.nx_str)]
        [x.append(i2) for i2 in range(self.ny_str) for _ in range(2)]

        # :allocate the output variables and couplings
        y = []
        [y.append(i3) for i3 in range(self.ny_str) for _ in range(4)]
        [y.append(i4) for i4 in range(self.nx_str)]

        # :define the structural submatrix
        aa = np.zeros(len(x) * len(y))
        percent = int(self.d_str * len(aa))
        aa[0:percent] = 1
        np.random.shuffle(aa)
        aa = aa.reshape(len(y), len(x))

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.matshow(aa, cmap='Purples')
        # plt.show()

        return aa

    def submatrix_structural_biased(self):
        """Add a bias in the distribution of the dots to test the accuracy of the scaled down matrix"""

        # :allocate the input variables and couplings
        x = []
        [x.append(i1) for i1 in range(self.nx_str_bias)]
        [x.append(i2) for i2 in range(self.ny_str_bias) for _ in range(2)]

        # :allocate the output variables and couplings
        y = []
        [y.append(i3) for _ in range(4) for i3 in range(self.ny_str_bias)]
        [y.append(i4) for i4 in range(self.nx_str_bias)]
        aa = np.zeros(len(x)*len(y))
        aa = aa.reshape(len(y), len(x))

        # :define the biased structural submatrix
        for j1 in range(4 * self.ny_str_bias + self.nx_str_bias):
            for i5 in range(0, 5):
                aa[j1][i5] = np.random.choice([1, 0], p=[self.d_str_bias, 1 - self.d_str_bias])
        for j2 in range(4 * self.ny_str_bias + self.nx_str_bias):
            for i6 in range(self.nx_str_bias + 2 * self.ny_str_bias - 5, self.nx_str_bias + 2 * self.ny_str_bias):
                aa[j2][i6] = np.random.choice([1, 0], p=[self.d_str_bias, 1 - self.d_str_bias])

        # int(4/5 * (self.nx_str_bias + 2 * self.ny_str_bias))
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.matshow(aa, cmap='Purples')
        # plt.show()

        return aa

    def submatrix_aerodynamics(self):
        """This is the aerodynamics part of the dependency matrix """

        # :allocate the input variables and couplings
        x = []
        [x.append(i1) for i1 in range(self.nx_aero)]
        [x.append(i2) for i2 in range(self.ny_aero) for _ in range(2)]

        # :allocate the output variables and couplings
        y = []
        [y.append(i3) for _ in range(4) for i3 in range(self.ny_aero)]
        [y.append(i4) for i4 in range(self.nx_aero)]

        # :define the aerodynamics submatrix
        aa = np.zeros(len(x) * len(y))
        percent = int(self.d_aero * len(aa))
        aa[0:percent] = 1
        np.random.shuffle(aa)
        aa = aa.reshape(len(y), len(x))
        return aa

    def submatrix_propulsion(self):
        """This is the propulsion part of the dependency matrix"""

        # :allocate the input variables and couplings
        x = []
        [x.append(i1) for i1 in range(self.nx_prop)]
        [x.append(i2) for i2 in range(self.ny_prop)]

        # :allocate the output variables and couplings
        y = []
        [y.append(i3) for _ in range(4) for i3 in range(self.ny_prop)]
        [y.append(i4) for i4 in range(self.nx_prop)]

        # :define the propulsion submatrix
        aa = np.zeros(len(x) * len(y))
        percent = int(self.d_prop * len(aa))
        aa[0:percent] = 1
        np.random.shuffle(aa)
        aa = aa.reshape(len(y), len(x))
        return aa

    def submatrix_shared_variables(self):
        """This is a part of the dependency matrix representing shared variables and all the components"""

        # :allocate the input variables and couplings
        x = []
        [x.append(i1) for i1 in range(self.nx_shared)]

        # :allocate the output variables and couplings
        y = []
        [y.append(i2) for _ in range(12) for i2 in range(self.ny_shared)]
        [y.append(i3) for _ in range(3) for i3 in range(self.nx_shared)]

        # :define the shared variables submatrix
        aa = np.zeros(len(x) * len(y))
        percent = int(self.d_shared * len(aa))
        aa[0:percent] = 1
        np.random.shuffle(aa)
        aa = aa.reshape(len(y), len(x))
        return aa


class ScaledDependencyMatrix:
    """
    This class creates component dependency matrix from a large random class with user defined
    nx and ny values for both random and scaled down matrix, along with the shift arguement for selecting
    the projection size
    """

    def __init__(self, nx_large, ny_large, structural, structural_biased, aerodynamics, propulsion, shared_variables):
        self.nx_large = nx_large
        self.ny_large = ny_large
        self.submatrix_structural = structural
        self.submatrix_structural_biased = structural_biased
        self.submatrix_aerodynamics = aerodynamics
        self.submatrix_propulsion = propulsion
        self.submatrix_shared_variables = shared_variables

    @staticmethod
    def build_component_dependency(nx_component, ny_component):
        """This method defines the component dependency graph"""

        # :create two dictionaries for coupling and constraints
        d_constraint = dict()
        d_coupling = dict()

        # :The original number of components in each constraint
        original_components = [7, 1, 4]

        # :The output labels for constraints are listed
        labels = ['g_1', 'g_2', 'g_3']

        if nx_component >= 7:
            # :The dependency graph is constructed
            dependency_graph = []
            for variable in range(np.size(original_components)):
                m, k, dependency = original_components[variable], [], []
                [k.append(1 + i1) for i1 in range(m)]
                while dependency == []:
                    for i2 in range(nx_component):
                        dependency.append(np.random.randint(1, m + 1))
                        if i2 == nx_component - 1 and set(k).issubset(set(dependency)):
                            break
                        elif i2 == nx_component - 1:
                            dependency = []
                            continue
                dependency_graph.append(dependency)

            # :The component dependency is named as a dictionary in the format "label: component"
            j1 = 0
            for i3 in dependency_graph:
                d_constraint[labels[j1]] = i3
                j1 += 1

        if nx_component < 7:
            # :The dependency graph is constructed
            dependency_graph = []
            for variable in range(np.size(original_components)):
                m, k, dependency = original_components[variable], [], []
                [k.append(1 + i1) for i1 in range(m)]
                dependency = []
                for i2 in range(nx_component):
                    dependency.append(np.random.randint(1, m + 1))
                dependency_graph.append(dependency)

            # :The component dependency is named as a dictionary in the format "label: component"
            j1 = 0
            for i3 in dependency_graph:
                d_constraint[labels[j1]] = i3
                j1 += 1

        # :The original number of components in each coupling output is listed
        original_components = [3, 1, 2, 2, 3, 1, 1, 1, 3, 1, 1, 1]

        # :The output labels are listed
        labels = ['y_1', 'y_11', 'y_12', 'y_14', 'y_2', 'y_21', 'y_23', 'y_24', 'y_3', 'y_31', 'y_32', 'y_34']

        if ny_component >= 3:
            # :The dependency graph is constructed
            dependency_graph = []
            for variable in range(np.size(original_components)):
                m, k, dependency = original_components[variable], [], []
                [k.append(1 + i1) for i1 in range(m)]
                while dependency == []:
                    for i2 in range(ny_component):
                        dependency.append(np.random.randint(1, m + 1))
                        if i2 == ny_component - 1 and set(k).issubset(set(dependency)):
                            break
                        elif i2 == ny_component - 1:
                            dependency = []
                            continue
                dependency_graph.append(dependency)

            # :The component dependency is named as a dictionary in the format "label: component"
            j1 = 0
            for i3 in dependency_graph:
                d_coupling[labels[j1]] = i3
                j1 += 1

        if ny_component < 3:
            # :The dependency graph is constructed
            dependency_graph = []
            for variable in range(np.size(original_components)):
                m, k, dependency = original_components[variable], [], []
                [k.append(1 + i1) for i1 in range(m)]
                dependency = []
                for i2 in range(ny_component):
                    dependency.append(np.random.randint(1, m + 1))
                dependency_graph.append(dependency)

            # :The component dependency is named as a dictionary in the format "label: component"
            j1 = 0
            for i3 in dependency_graph:
                d_coupling[labels[j1]] = i3
                j1 += 1

        return dict(d_constraint.items() + d_coupling.items())

    @staticmethod
    def label(nx_label, ny_label):
        """Set the input and output labels for the scaled matrix"""

        #: define input variable and coupling labels
        label_x = []
        label_x.append('x_1'), [label_x.append('') for _ in range(nx_label - 1)]
        label_x.append('y_21'), [label_x.append('') for _ in range(ny_label - 1)]
        label_x.append('y_31'), [label_x.append('') for _ in range(ny_label - 1)]
        label_x.append('x_2'), [label_x.append('') for _ in range(nx_label - 1)]
        label_x.append('y_12'), [label_x.append('') for _ in range(ny_label - 1)]
        label_x.append('y_32'), [label_x.append('') for _ in range(ny_label - 1)]
        label_x.append('x_3'), [label_x.append('') for _ in range(nx_label - 1)]
        label_x.append('y_23'), [label_x.append('') for _ in range(ny_label - 1)]
        label_x.append('x_shared'), [label_x.append('') for _ in range(nx_label - 1)]

        # : define output coupling labels
        label_y = []
        label_y.append('y_1'), [label_y.append('') for _ in range(ny_label - 1)]
        label_y.append('y_11'), [label_y.append('') for _ in range(ny_label - 1)]
        label_y.append('y_12'), [label_y.append('') for _ in range(ny_label - 1)]
        label_y.append('y_14'), [label_y.append('') for _ in range(ny_label - 1)]
        label_y.append('g_1'), [label_y.append('') for _ in range(nx_label - 1)]
        label_y.append('y_2'), [label_y.append('') for _ in range(ny_label - 1)]
        label_y.append('y_21'), [label_y.append('') for _ in range(ny_label - 1)]
        label_y.append('y_23'), [label_y.append('') for _ in range(ny_label - 1)]
        label_y.append('y_24'), [label_y.append('') for _ in range(ny_label - 1)]
        label_y.append('g_2'), [label_y.append('') for _ in range(nx_label - 1)]
        label_y.append('y_3'), [label_y.append('') for _ in range(ny_label - 1)]
        label_y.append('y_31'), [label_y.append('') for _ in range(ny_label - 1)]
        label_y.append('y_32'), [label_y.append('') for _ in range(ny_label - 1)]
        label_y.append('y_34'), [label_y.append('') for _ in range(ny_label - 1)]
        label_y.append('g_3'), [label_y.append('') for _ in range(nx_label - 1)]

        return label_x, label_y

    def scaled_dependency_matrix(self, nx_scale, ny_scale):
        """Scale down the dependency matrix using a fixed ny value for the large random matrix"""
        nx2 = self.nx_large
        ny2 = self.ny_large

        # :shift size for each place in the scaled compoenent
        shift = 1

        # : scaling down 1st dependency matrix
        a1 = self.submatrix_structural  #: add bias for checking the bias
        kx = nx2 + 2 * ny2 - shift * (nx_scale + 2 * ny_scale - 1)
        b1 = np.zeros((4 * ny_scale + nx_scale, nx_scale + 2 * ny_scale))
        for i1 in range(nx_scale + 2 * ny_scale):
            for j1 in range(4 * ny_scale + nx_scale):
                if j1 in np.arange(0, ny_scale):
                    j2 = np.random.choice(np.arange(0, ny2))
                    probability = np.average(a1[j2][i1 * shift: i1 * shift + kx])
                    b1[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(ny_scale, 2 * ny_scale):
                    j2 = np.random.choice(np.arange(ny2, 2 * ny2))
                    probability = np.average(a1[j2][i1 * shift: i1 * shift + kx])
                    b1[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(2 * ny_scale, 3 * ny_scale):
                    j2 = np.random.choice(np.arange(2 * ny2, 3 * ny2))
                    probability = np.average(a1[j2][i1 * shift: i1 * shift + kx])
                    b1[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(3 * ny_scale, 4 * ny_scale):
                    j2 = np.random.choice(np.arange(3 * ny2, 4 * ny2))
                    probability = np.average(a1[j2][i1 * shift: i1 * shift + kx])
                    b1[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(4 * ny_scale, 4 * ny_scale + nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2, 4 * ny2 + nx2))
                    probability = np.average(a1[j2][i1 * shift: i1 * shift + kx])
                    b1[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])

        # : scaling down 2nd dependency matrix
        a2 = self.submatrix_aerodynamics
        kx = nx2 + 2 * ny2 - shift * (nx_scale + 2 * ny_scale - 1)
        b2 = np.zeros((4 * ny_scale + nx_scale, nx_scale + 2 * ny_scale))
        for i1 in range(nx_scale + 2 * ny_scale):
            for j1 in range(4 * ny_scale + nx_scale):
                if j1 in np.arange(0, ny_scale):
                    j2 = np.random.choice(np.arange(0, ny2))
                    probability = np.average(a2[j2][i1 * shift: i1 * shift + kx])
                    b2[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(ny_scale, 2 * ny_scale):
                    j2 = np.random.choice(np.arange(ny2, 2 * ny2))
                    probability = np.average(a2[j2][i1 * shift: i1 * shift + kx])
                    b2[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(2 * ny_scale, 3 * ny_scale):
                    j2 = np.random.choice(np.arange(2 * ny2, 3 * ny2))
                    probability = np.average(a2[j2][i1 * shift: i1 * shift + kx])
                    b2[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(3 * ny_scale, 4 * ny_scale):
                    j2 = np.random.choice(np.arange(3 * ny2, 4 * ny2))
                    probability = np.average(a2[j2][i1 * shift: i1 * shift + kx])
                    b2[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(4 * ny_scale, 4 * ny_scale + nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2, 4 * ny2 + nx2))
                    probability = np.average(a2[j2][i1 * shift: i1 * shift + kx])
                    b2[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])

        # : scaling down 3rd dependency matrix
        a3 = self.submatrix_propulsion
        kx = nx2 + 1 * ny2 - shift * (nx_scale + 1 * ny_scale - 1)
        b3 = np.zeros((4 * ny_scale + nx_scale, nx_scale + 1 * ny_scale))
        for i1 in range(nx_scale + 1 * ny_scale):
            for j1 in range(4 * ny_scale + nx_scale):
                if j1 in np.arange(0, ny_scale):
                    j2 = np.random.choice(np.arange(0, ny2))
                    probability = np.average(a3[j2][i1 * shift: i1 * shift + kx])
                    b3[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(ny_scale, 2 * ny_scale):
                    j2 = np.random.choice(np.arange(ny2, 2 * ny2))
                    probability = np.average(a3[j2][i1 * shift: i1 * shift + kx])
                    b3[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(2 * ny_scale, 3 * ny_scale):
                    j2 = np.random.choice(np.arange(2 * ny2, 3 * ny2))
                    probability = np.average(a3[j2][i1 * shift: i1 * shift + kx])
                    b3[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(3 * ny_scale, 4 * ny_scale):
                    j2 = np.random.choice(np.arange(3 * ny2, 4 * ny2))
                    probability = np.average(a3[j2][i1 * shift: i1 * shift + kx])
                    b3[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(4 * ny_scale, 4 * ny_scale + nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2, 4 * ny2 + nx2))
                    probability = np.average(a3[j2][i1 * shift: i1 * shift + kx])
                    b3[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])

        # :scaling down 4th dependency matrix
        a4 = self.submatrix_shared_variables
        kx = nx2 - shift * (nx_scale - 1)
        b4 = np.zeros((12 * ny_scale + 3 * nx_scale, nx_scale))
        for i1 in range(nx_scale):
            for j1 in range(12 * ny_scale + 3 * nx_scale):
                if j1 in np.arange(0, ny_scale):
                    j2 = np.random.choice(np.arange(0, ny2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(ny_scale, 2 * ny_scale):
                    j2 = np.random.choice(np.arange(ny2, 2 * ny2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(2 * ny_scale, 3 * ny_scale):
                    j2 = np.random.choice(np.arange(2 * ny2, 3 * ny2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(3 * ny_scale, 4 * ny_scale):
                    j2 = np.random.choice(np.arange(3 * ny2, 4 * ny2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(4 * ny_scale, 4 * ny_scale + nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2, 4 * ny2 + nx2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(4 * ny_scale + nx_scale, 5 * ny_scale + nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2 + nx2, 4 * ny2 + nx2 + ny2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(5 * ny_scale + nx_scale, 6 * ny_scale + nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2 + nx2 + ny2, 4 * ny2 + nx2 + 2 * ny2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(6 * ny_scale + nx_scale, 7 * ny_scale + nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2 + nx2 + 2 * ny2, 4 * ny2 + nx2 + 3 * ny2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(7 * ny_scale + nx_scale, 8 * ny_scale + nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2 + nx2 + 3 * ny2, 4 * ny2 + nx2 + 4 * ny2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(8 * ny_scale + nx_scale, 8 * ny_scale + 2 * nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2 + nx2 + 4 * ny2, 4 * ny2 + nx2 + 4 * ny2 + nx2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(8 * ny_scale + 2 * nx_scale, 9 * ny_scale + 2 * nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2 + nx2 + 4 * ny2 + nx2, 4 * ny2 + nx2 + 4 * ny2 + nx2 + ny2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(9 * ny_scale + 2 * nx_scale, 10 * ny_scale + 2 * nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2 + nx2 + 4 * ny2 + nx2 + ny2, 4 * ny2 + nx2 + 4 * ny2 + nx2 +
                                                    2 * ny2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(10 * ny_scale + 2 * nx_scale, 11 * ny_scale + 2 * nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2 + nx2 + 4 * ny2 + nx2 + 2 * ny2, 4 * ny2 + nx2 + 4 * ny2 +
                                                    nx2 + 3 * ny2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(11 * ny_scale + 2 * nx_scale, 12 * ny_scale + 2 * nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2 + nx2 + 4 * ny2 + nx2 + 3 * ny2, 4 * ny2 + nx2 + 4 * ny2 +
                                                    nx2 + 4 * ny2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])
                elif j1 in np.arange(12 * ny_scale + 2 * nx_scale, 12 * ny_scale + 3 * nx_scale):
                    j2 = np.random.choice(np.arange(4 * ny2 + nx2 + 4 * ny2 + nx2 + 4 * ny2, 4 * ny2 + nx2 + 4 * ny2 +
                                                    nx2 + 4 * ny2 + nx2))
                    probability = np.average(a4[j2][i1 * shift: i1 * shift + kx])
                    b4[j1][i1] = np.random.choice([1, 0], p=[probability, 1 - probability])

        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # ax.set_xlabel('large random dependency matrix', fontsize=18)

        aa_1 = bd(a1, a2, a3)
        aa_2 = np.concatenate((aa_1, a4), axis=1)
        # cax1 = ax.matshow(aa_2,cmap = 'Purples')

        # ax = fig.add_subplot(122)
        # ax.set_xlabel('scaled dependency matrix', fontsize=18)
        bb_1 = bd(b1, b2, b3)
        bb_2 = np.concatenate((bb_1, b4), axis=1)
        # cax = ax.matshow(bb_2, cmap='Purples')

        # plt.show()

        return bb_2


# :visualize large random matrix
# fig = plt.figure(1)
# ax1 = fig.add_subplot(121)
# [nx_large, ny_large, d_large] = pickle.load(open("large_matrix_params.p", "rb"))
# v1 = LargeRandomMatrix(nx_large, ny_large, d_large)
# submatrix_structural = v1.submatrix_structural()
# submatrix_structural_biased = v1.submatrix_structural_biased()
# submatrix_aerodynamics = v1.submatrix_aerodynamics()
# submatrix_propulsion = v1.submatrix_propulsion()
# submatrix_shared_variables = v1.submatrix_shared_variables()
# ax1.set_xlabel('large_dependency matrix' + ' density:' + str(d_large), fontsize=10)
# aa_1 = bd(submatrix_structural,
#           submatrix_aerodynamics,
#           submatrix_propulsion)
# aa_2 = submatrix_shared_variables
# aa_3 = np.concatenate((aa_1, aa_2), axis=1)
# ax1.matshow(aa_3, cmap='Purples')
# # : generate scaled dependency matrix
# nx = int(pickle.load(open("input_values.p", "rb"))[0])
# ny = int(pickle.load(open("input_values.p", "rb"))[1])
# a = 0
# while a != 12 * ny + 3 * nx:
#     v2 = ScaledDependencyMatrix(nx_large=nx_large,
#                                 ny_large=ny_large,
#                                 structural=submatrix_structural,
#                                 structural_biased=submatrix_structural_biased,
#                                 aerodynamics=submatrix_aerodynamics,
#                                 propulsion=submatrix_propulsion,
#                                 shared_variables=submatrix_shared_variables
#                                 )
#     scaled_matrix = v2.scaled_dependency_matrix(nx, ny)
#     a = 0
#     for i in range(12 * ny + 3 * nx):
#         if 1 in scaled_matrix[i]:
#             a += 1
# label = v2.label(nx, ny)
# fig.suptitle('(nx,ny) : ' + '(' + str(nx) + ',' + str(ny) + ')', fontsize=18)
# ax2 = fig.add_subplot(122)
# ax2.matshow(scaled_matrix, cmap='Purples')
# ax2.set_xlabel('scaled_dependency matrix' + ' density:' + str(v1.d_str), fontsize=10)
# ax2.set_xticklabels(label[0], fontdict={'fontsize': 7, 'rotation': 90})
# plt.xticks(np.arange(0, 4 * nx + 5 * ny, 1))
# ax2.set_yticklabels(label[1], fontdict={'fontsize': 7})
# plt.yticks(np.arange(0, 3 * nx + 12 * ny, 1))
# fig.tight_layout()
# plt.show()

# # # set  the value of nx for the scaled down matrix
# # nx = 2
# # fig = plt.figure(3)
# # fig.suptitle('(nx,ny) : ' + '(' + str(nx) + ',' + str(nx) + ')', fontsize=18)
# # ax = fig.add_subplot(231)
# # ax.set_xlabel('large random dependency matrix' + ' density: 0.4', fontsize=14)
# # a = v1.scaled_dependency_matrix(nx, nx, .4)
# # ax.matshow(a[1], cmap='Purples')
# # j = 0
# # for i in [.2, .4, .6, .8, 1]:
# #     ax = fig.add_subplot(int('23' + str(j + 2)))
# #     ax.set_xlabel('scaled dependency matrix' + '  density:' + str(i), fontsize=14)
# #     b = v1.scaled_dependency_matrix(nx, nx, i)
# #     ax.matshow(b[0], cmap='Purples')
# #     j += 1
# #
# #
# # nx = 7
# # fig = plt.figure(4)
# # fig.suptitle('(nx,ny) : ' + '(' + str(nx) + ',' + str(nx) + ')', fontsize=18)
# # ax = fig.add_subplot(231)
# # ax.set_xlabel('large random dependency matrix' + ' density: 0.4', fontsize=14)
# # a = v1.scaled_dependency_matrix(nx, nx, .4)
# # ax.matshow(a[1], cmap='Purples')
# # j = 0
# # for i in [.2, .4, .6, .8, 1]:
# #     ax = fig.add_subplot(int('23' + str(j + 2)))
# #     ax.set_xlabel('scaled dependency matrix' + '  density:' + str(i), fontsize=14)
# #     b = v1.scaled_dependency_matrix(dict(a.items() + b.items())nx, nx, i)
# #     ax.matshow(b[0], cmap='Purples')
# #     j += 1
# plt.show()


# a = ScaledDependencyMatrix(1, 1, 1, 1, 1, 1, 1)
# print(a.build_component_dependency(5, 8))