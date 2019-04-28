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
from examples.scripts.bounds_initialization import create_bounds_dict
from examples.scripts.ssbj_vanaret_idf2 import idf_run2
from examples.scripts.vanaret import ScaledDependencyMatrix, LargeRandomMatrix
from examples.scripts.ssbj_vanaret_idf import idf_run
from examples.scripts.ssbj_vanaret_mdf import mdf_run
import time

# # : enter and pickle values for the large dependency matrix
nx_large = 40
ny_large = 2
d_large = .5
# pickle.dump([nx_large, ny_large, d_large], open("large_matrix_params.p", "wb"))
# v1 = LargeRandomMatrix(nx_large, ny_large, d_large)
#
# # : pickle the submatrices of the larger matrix
# pickle.dump(v1.submatrix_structural(), open("submatrix_structural.p", "wb"))  # pickle the input values
# pickle.dump(v1.submatrix_structural_biased(), open("submatrix_structural_biased.p", "wb"))  # pickle the input values
# pickle.dump(v1.submatrix_aerodynamics(), open("submatrix_aerodynamics.p", "wb"))  # pickle the input values
# pickle.dump(v1.submatrix_propulsion(), open("submatrix_propulsion.p", "wb"))  # pickle the input values
# pickle.dump(v1.submatrix_shared_variables(), open("submatrix_shared_variables.p", "wb"))  # pickle the input values


#: make an empty pandas dataframe to form database
if not os.path.isfile('df_mdf.p'):
    pickle.dump(pd.DataFrame(columns=['nx', 'ny', 'total iterations[MDF]', 'total time[MDF]', 'final_objective[MDF]']),
                open("df_mdf.p", "wb"))
if not os.path.isfile('df_idf.p'):
    pickle.dump(pd.DataFrame(columns=['total iterations[IDF]', 'total time[IDF]', 'final_objective[IDF]']),
                open("df_idf.p", "wb"))
if not os.path.isfile('df_idf_tol.p'):
    pickle.dump(pd.DataFrame(columns=['total iterations[IDF_tol]', 'total time[IDF_tol]', 'final_objective[IDF_tol]']),
                open("df_idf_tol.p", "wb"))


for i in range(10, 15):
    # # : enter the inputs for problem size and coupling density
    nx = 7  # dimension of discipline inputs
    ny = i  # dimension of coupling
    user_input_array = [nx, ny, d_large]
    pickle.dump(user_input_array, open("input_values.p", "wb"))  # pickle the input values

    # : enter the inputs for constraint parameters
    # [p_g1, p_g2, p_g3] = [], [], []  # the percentage of active constraints at initial point
    # [p_g1.append(np.random.random_sample()) for _ in range(nx)]
    # [p_g2.append(np.random.random_sample()) for _ in range(nx)]
    # [p_g3.append(np.random.random_sample()) for _ in range(nx)]
    # alpha_g1, alpha_g2, alpha_g3 = [], [], []  # alpha determines to what extent inactive constraints are satisfied
    # [alpha_g1.append(np.random.random_sample()) for _ in range(nx)]
    # [alpha_g2.append(np.random.random_sample()) for _ in range(nx)]
    # [alpha_g3.append(np.random.random_sample()) for _ in range(nx)]
    # mu_g1, mu_g2, mu_g3 = [], [], []  # select random number in (0,1 ) with uniform probability to activate constraint
    # [mu_g1.append(np.random.random_sample()) for _ in range(nx)]
    # [mu_g2.append(np.random.random_sample()) for _ in range(nx)]
    # [mu_g3.append(np.random.random_sample()) for _ in range(nx)]
    # constraint_parameters = ([p_g1, p_g2, p_g3], [alpha_g1, alpha_g1, alpha_g1], [mu_g1, mu_g2, mu_g3])
    # pickle.dump(constraint_parameters, open("constraint_params.p", "wb"))  # pickle the input values

    # initialize the class which build the dependency matrix
    v2 = ScaledDependencyMatrix(nx_large=nx_large,
                                ny_large=ny_large,
                                structural=pickle.load(open("submatrix_structural.p", "rb")),
                                structural_biased=pickle.load(open("submatrix_structural_biased.p", "rb")),
                                aerodynamics=pickle.load(open("submatrix_aerodynamics.p", "rb")),
                                propulsion=pickle.load(open("submatrix_propulsion.p", "rb")),
                                shared_variables=pickle.load(open("submatrix_shared_variables.p", "rb"))
                                )

    component_dependency = v2.build_component_dependency(nx, ny)  # remap the extrapolations to original components
    # print(component_dependency)
    dependency_matrix = v2.scaled_dependency_matrix(nx, ny)  # dependency matrix to determine the coupling density
    pickle.dump(component_dependency, open("component_dependency.p", "wb"))  # pickle the component dependency graph
    pickle.dump(dependency_matrix, open("dependency_matrix.p", "wb"))  # pickle the dependency matrix
    #
    # Display the scaled down matrix
    # fig = plt.figure(1)
    # ax1 = fig.add_subplot(121)
    # ax1.matshow(v2.scaled_dependency_matrix(nx, ny), cmap='Purples')
    # plt.show()

    # Initialize and pickle the  bounds for the original component output
    create_bounds_dict()

    # supply the MDAO definitions
    mdao_definitions = ['MDF', 'IDF']

    # Optimization problem
    if 'IDF' in mdao_definitions:
        idf_run2(nx, ny)
    if 'MDF' in mdao_definitions:
        mdf_run(nx, ny, d_large)

    df_mdf = pickle.load(open("df_mdf.p", "rb"))
    df_idf = pickle.load(open("df_idf.p", "rb"))
    df_idf_tol = pickle.load(open("df_idf_tol.p", "rb"))

    result = pd.concat([df_mdf, df_idf, df_idf_tol], axis=1, join='inner')
    result.to_csv('result2.csv')
