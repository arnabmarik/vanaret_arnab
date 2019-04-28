from __future__ import print_function
import pickle
import random

import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pylab as plt
from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
from ssbj_vanaret_mda import SsbjMda
import time


def mdf_run(nx, ny, d):
    prob = Problem()  # initialize the optimization problem
    prob.model = SsbjMda(nx_input=nx)  # create the MDA

    # Design variables
    prob.model.add_design_var('z', lower=np.zeros(nx), upper=np.ones(nx))  # shared variables
    prob.model.add_design_var('x1', lower=np.zeros(nx), upper=np.ones(nx))  # local variables for structural discipline
    prob.model.add_design_var('x2', lower=np.zeros(nx),
                              upper=np.ones(nx))  # local variables for aerodynamics discipline
    prob.model.add_design_var('x3', lower=np.zeros(nx), upper=np.ones(nx))  # local variables for propulsion discipline

    # Objective function
    prob.model.add_objective('range')

    # Constraints
    for i in range(nx):
        prob.model.add_constraint('con_g1' + str(i + 1), upper=0)
        prob.model.add_constraint('con_g2' + str(i + 1), upper=0)
        prob.model.add_constraint('con_g3' + str(i + 1), upper=0)

    prob.driver = ScipyOptimizeDriver(optimizer='SLSQP')
    prob.driver.add_recorder(SqliteRecorder("cases_mdf.sql"))
    prob.driver.options['maxiter'] = 30  # random.randint(20, 30)
    prob.driver.options['tol'] = 1e-3

    start_time = time.time()
    prob.setup(mode='fwd')
    prob.set_solver_print(1)
    prob.run_driver()
    end_time = time.time()
    prob.cleanup()
    total_time = end_time - start_time

    iters = len(CaseReader('cases_mdf.sql').get_cases())
    case_ids = CaseReader('cases_mdf.sql').get_cases()
    obj_list = ['range']
    z = []
    [z.append(case.get_objectives(case)[obj_list[0]]) for case in case_ids]
    df_mdf = pickle.load(open("df_mdf.p", "rb")).append(pd.DataFrame({'nx': [nx], 'ny': [ny], 'd': [d],
                                                                      'total iterations[MDF]': [iters],
                                                                      'total time[MDF]': [total_time],
                                                                      'final_objective[MDF]': z[-1]}))
    pickle.dump(df_mdf, open("df_mdf.p", "wb"))
    print('z_opt=', prob['z'])
    print('x1_opt=', prob['x1'])
    print('x2_opt=', prob['x2'])
    print('x3_opt=', prob['x3'])

# #---------------------------------------------------------------------------------------------------------------- #
# #         The following section generates the constraint and the objective convergence graphs                     #
# #---------------------------------------------------------------------------------------------------------------- #

    # history of constraints for MDF
    # cr = CaseReader('cases_mdf.sql')
    # case_ids = cr.get_cases()
    # total = []
    # constraints_list = ['con_g1', 'con_g2', 'con_g2']
    # for k in range(3):
    #     for i in range(nx):
    #         z = []
    #         [z.append(case.get_constraints(case)[constraints_list[k] + str(i + 1)]) for case in case_ids]
    #         total.append(z)
    # b = np.array(total).reshape(nx * 3, len(case_ids))
    # pickle.dump(b, open("con_mdf.p", "wb"))
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # a = ax.pcolor(b, cmap='viridis')
    # plt.xlabel('MDF driver cases')
    # plt.ylabel('Constraints')
    # label = []
    # for i in range(3):
    #     for j in range(nx):
    #         label.append('g' + str(i + 1) + '_' + str(j + 1))
    # ax.set_yticklabels(label)
    # plt.yticks(np.arange(0, 3 * nx, 1))
    # fig.colorbar(a)
    #
    # # objective function history for MDF`
    # cr = CaseReader('cases_mdf.sql')
    # case_ids = cr.get_cases()
    # obj_list = ['range']
    # z = []
    # [z.append(case.get_objectives(case)[obj_list[0]]) for case in case_ids]
    # pickle.dump(z, open("obj_mdf.p", "wb"))
    # fig = plt.figure(2)
    # ax = fig.add_subplot(111)
    # a = ax.plot(z)
    # plt.xlabel('MDF driver cases')
    # plt.ylabel('objective')
    #
    # # norm of the residuals
    # cr = CaseReader('cases_mdf.sql')
    # cases = cr.get_cases()
    # z = []
    # [z.append(case.get_design_vars(case)['z']) for case in cases]
    # x1 = []
    # [x1.append(case.get_design_vars(case)['x1']) for case in cases]
    # x2 = []
    # [x2.append(case.get_design_vars(case)['x2']) for case in cases]
    # x3 = []
    # [x3.append(case.get_design_vars(case)['x3']) for case in cases]
    #
    # sum_z, sum_x1, sum_x2, sum_x3 = [], [], [], []
    # for i in range(len(cases) - 1):
    #     sum_z.append(np.sum(np.subtract(z[i], z[i + 1]) ** 2))
    #     sum_x1.append(np.sum(np.subtract(x1[i], x1[i + 1]) ** 2))
    #     sum_x2.append(np.sum(np.subtract(x2[i], x2[i + 1]) ** 2))
    #     sum_x3.append(np.sum(np.subtract(x3[i], x3[i + 1]) ** 2))
    # d_sum = []
    # for i in range(len(cases) - 1):
    #     d_sum.append(sum_x1[i] + sum_x2[i] + sum_x3[i] + sum_z[i])
    # pickle.dump(d_sum, open("l2_norm_xdes_mdf.p", "wb"))
    # fig = plt.figure(3)
    # ax = fig.add_subplot(111)
    # a = ax.plot(d_sum)
    # plt.xlabel('driver cases')
    # plt.ylabel('d_sum')
    # # plt.show()
