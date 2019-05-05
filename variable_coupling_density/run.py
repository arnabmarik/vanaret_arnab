"""
This script generates the dependency matrix and the component dependency graph, followed by MDF and IDF optimization
routines on the SSBJ Problem. This is made in accordance with the method used in the paper: "On the Consequences of the
No Free Lunch" Theorem for Optimization on the Choice of an Appropriate MDO Architecture"
"""
# Imports
from __future__ import print_function
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from bounds_initialization import create_bounds_dict
from ssbj_vanaret_idf2 import idf_run2
from ssbj_vanaret_mdf import mdf_run
from vanaret import ScaledDependencyMatrix, LargeRandomMatrix

for i in [0.3, 0.7]:
    for j in [0.3, 0.7]:
        for k in [0.3, 0.7]:
            for l in [0.4, 0.7]:
                for d_large in [(i, j, k, l)]:
                    nx_large = 100
                    ny_large = 2

                    with open("large_matrix_params.p", "wb") as f:
                        pickle.dump([nx_large, ny_large, d_large], f)
                    v1 = LargeRandomMatrix(nx_large, ny_large, d_large)

                    # pickle the submatrices of the larger matrix
                    a = ["submatrix_structural", "submatrix_structural_biased", "submatrix_aerodynamics",
                         "submatrix_propulsion",
                         "submatrix_shared_variables"]
                    for i in a:
                        with open(i + ".p", "wb") as f:
                            pickle.dump(getattr(v1, i)(), f)

                    # make an empty pandas dataframe to form database
                    a = ["df_mdf.p", "df_idf.p", "df_idf_tol.p"]
                    for i in a:
                        if not os.path.isfile(i):
                            with open(i, "wb") as f:
                                pickle.dump(pd.DataFrame(columns=[]), f)

                    for nx in [4]:
                        for ny in [4]:
                            for _ in range(1):
                                # nx = 5  # dimension of discipline inputs
                                # ny = j  # dimension of coupling
                                user_input_array = [nx, ny, d_large]
                                with open("input_values.p", "wb") as f:
                                    pickle.dump(user_input_array, f)  # pickle the input values

                                # enter the inputs for constraint parameters
                                [p_g1, p_g2, p_g3] = [], [], []  # the percentage of active constraints at initial point
                                [p_g1.append(np.random.random_sample()) for _ in range(nx)]
                                [p_g2.append(np.random.random_sample()) for _ in range(nx)]
                                [p_g3.append(np.random.random_sample()) for _ in range(nx)]
                                alpha_g1, alpha_g2, alpha_g3 = [], [], []  # determines to what extent inactive constraint are satisfied
                                [alpha_g1.append(np.random.random_sample()) for _ in range(nx)]
                                [alpha_g2.append(np.random.random_sample()) for _ in range(nx)]
                                [alpha_g3.append(np.random.random_sample()) for _ in range(nx)]
                                mu_g1, mu_g2, mu_g3 = [], [], []  # random number in (0,1 ) with uniform probability to activ. comstr.
                                [mu_g1.append(np.random.random_sample()) for _ in range(nx)]
                                [mu_g2.append(np.random.random_sample()) for _ in range(nx)]
                                [mu_g3.append(np.random.random_sample()) for _ in range(nx)]
                                constraint_parameters = (
                                [p_g1, p_g2, p_g3], [alpha_g1, alpha_g1, alpha_g1], [mu_g1, mu_g2, mu_g3])
                                with open("constraint_params.p", "wb") as f:
                                    pickle.dump(constraint_parameters, f)
                                # initialize the class which build the dependency matrix
                                print("generating valid dependency matrix")
                                a = 0
                                while a != 12 * ny + 3 * nx:  # : need to ensure a matrix is formed with no empty rows
                                    disc = ["structural", "structural_biased", "aerodynamics", "propulsion",
                                            "shared_variables"]
                                    k = []
                                    for i in disc:
                                        with open("submatrix_" + i + ".p", "rb") as f:
                                            k.append(pickle.load(f))
                                    v2 = ScaledDependencyMatrix(nx_large=nx_large,
                                                                ny_large=ny_large,
                                                                structural=k[0],
                                                                structural_biased=k[1],
                                                                aerodynamics=k[2],
                                                                propulsion=k[3],
                                                                shared_variables=k[4]
                                                                )
                                    dependency_matrix = v2.scaled_dependency_matrix(nx, ny)
                                    a = 0
                                    for h in range(12 * ny + 3 * nx):
                                        if 1 in dependency_matrix[h]:
                                            a += 1
                                print("done!")
                                component_dependency = v2.build_component_dependency(nx, ny)
                                with open("component_dependency.p", "wb") as f:
                                    pickle.dump(component_dependency, f)  # pickle component dependency graph
                                with open("dependency_matrix.p", "wb") as f:
                                    pickle.dump(dependency_matrix, f)  # pickle the dependency matrix

                                # Display the scaled down matrix
                                fig = plt.figure(1)
                                ax1 = fig.add_subplot(111)
                                label = v2.label(nx, ny)
                                ax1.matshow(dependency_matrix, cmap='Purples')
                                ax1.set_xlabel('scaled_dependency matrix' + ' density:' + str(v1.d_str), fontsize=10)
                                ax1.set_xticklabels(label[0], fontdict={'fontsize': 7, 'rotation': 90})
                                plt.xticks(np.arange(0, 4 * nx + 5 * ny, 1))
                                ax1.set_yticklabels(label[1], fontdict={'fontsize': 7})
                                plt.yticks(np.arange(0, 3 * nx + 12 * ny, 1))
                                fig.tight_layout()
                                # plt.show()

                                # Initialize and pickle the  bounds for the original component output
                                create_bounds_dict()

                                # supply the MDAO definitions
                                mdao_definitions = ['MDF', 'IDF']

                                if 'IDF' in mdao_definitions:
                                    idf_run2(nx, ny)

                                if 'MDF' in mdao_definitions:
                                    mdf_run(nx, ny, d_large)

                                df_mdf = pickle.load(open("df_mdf.p", "rb"))
                                df_idf = pickle.load(open("df_idf.p", "rb"))
                                df_idf_tol = pickle.load(open("df_idf_tol.p", "rb"))

                                if not os.path.exists('./result/result(nx = ' + str(nx) + ')'):
                                    os.makedirs('./result/result(nx = ' + str(nx) + ')')

                                result = pd.concat([df_mdf, df_idf_tol], axis=1)
                                result.to_csv('result/result(nx = ' + str(nx) + ')/result_new_values' + '('
                                              + str(d_large) + ').csv')






