from __future__ import print_function
import pickle
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder
import time
import random
from openmdao.recorders.case_reader import CaseReader

from ssbj_vanaret_mda import SsbjMda
from ssbj_vanaret_idf2_mda import SsbjIdf2Mda



def idf_run2(nx, ny):

    # make a counter for discipline calls
    a = ["str_count.p", "aer_count.p", "pro_count.p", "obj_count.p"]
    for i in a:
        with open(i, "wb") as f:
            pickle.dump(0, f)

    # Initialize an MDA to generate the starting point for IDF
    prob_init = Problem()  # initialize the optimization problem
    prob_init.model = SsbjMda(nx_input=nx)  # create the MDA

    # Design variables
    prob_init.model.add_design_var('z',  lower=np.zeros(nx), upper=np.ones(nx))
    prob_init.model.add_design_var('x1', lower=np.zeros(nx), upper=np.ones(nx))
    prob_init.model.add_design_var('x2', lower=np.zeros(nx), upper=np.ones(nx))
    prob_init.model.add_design_var('x3', lower=np.zeros(nx), upper=np.ones(nx))

    # Objective function
    prob_init.model.add_objective('range')

    # Constraints
    for i in range(nx):
        prob_init.model.add_constraint('con_g1' + str(i + 1), upper=0)
        prob_init.model.add_constraint('con_g2' + str(i + 1), upper=0)
        prob_init.model.add_constraint('con_g3' + str(i + 1), upper=0)

    prob_init.driver = ScipyOptimizeDriver(optimizer='SLSQP')
    prob_init.driver.options['maxiter'] = 0

    prob_init.setup(mode='fwd')
    prob_init.set_solver_print(1)
    prob_init.run_driver()
    prob_init.cleanup()
    y12_initial = prob_init['y12']
    y23_initial = prob_init['y23']
    y32_initial = prob_init['y32']
    y21_initial = prob_init['y21']
    y31_initial = prob_init['y31']

    # : initialize MDA for IDF
    prob = Problem()
    prob.model = SsbjIdf2Mda(nx, ny, y12_initial, y23_initial, y32_initial, y21_initial, y31_initial)
    # create the MDA
    # Design variables
    prob.model.add_design_var('z', lower=np.zeros(nx), upper=np.ones(nx))  # shared variables
    prob.model.add_design_var('x1', lower=np.zeros(nx), upper=np.ones(nx))  # local variable for structural discipline
    prob.model.add_design_var('x2', lower=np.zeros(nx), upper=np.ones(nx))  # local variable for aerodynamic discipline
    prob.model.add_design_var('x3', lower=np.zeros(nx), upper=np.ones(nx))  # local variable for propulsion discipline

    # # coupling variables
    prob.model.add_design_var('y31')
    prob.model.add_design_var('y12')
    prob.model.add_design_var('y32')
    prob.model.add_design_var('y23')
    prob.model.add_design_var('y21')

    # Objective function
    prob.model.add_objective('obj')

    # Constraints
    for i in range(nx):
        prob.model.add_constraint('con_g1' + str(i + 1), upper=0)
        prob.model.add_constraint('con_g2' + str(i + 1), upper=0)
        prob.model.add_constraint('con_g3' + str(i + 1), upper=0)

    epsilon = 1e-9
    # Coupling constraints
    for i in range(ny):
        prob.model.add_constraint('con_y12' + str(i + 1), upper=epsilon)
        prob.model.add_constraint('con_y21' + str(i + 1), upper=epsilon)
        prob.model.add_constraint('con_y23' + str(i + 1), upper=epsilon)
        prob.model.add_constraint('con_y32' + str(i + 1), upper=epsilon)
        prob.model.add_constraint('con_y31' + str(i + 1), upper=epsilon)

    # Optimizer options
    prob.driver = ScipyOptimizeDriver()
    prob.set_solver_print(2)
    prob.driver.options['optimizer'] = 'SLSQP'
    for tol in [1e-3]:
        prob.driver.options['maxiter'] = random.randint(40, 50)
        prob.driver.options['tol'] = tol
        prob.driver.add_recorder(SqliteRecorder("cases_idf.sql"))
        # Run optimization
        start_time = time.time()
        prob.setup(mode='fwd')

        # view_model(prob, outfile='n2_mdfgs.html', show_browser=True)
        prob.run_driver()
        prob.run_model()
        # prob.check_partials()
        prob.cleanup()
        end_time = time.time()
        total_time = end_time - start_time
        if prob.driver.options['tol'] == 1e-6:
            iters = len(CaseReader('cases_idf.sql').get_cases())
            cr = CaseReader('cases_idf.sql')
            case_ids = cr.get_cases()
            obj_list = ['obj']
            z = []
            [z.append(case.get_objectives(case)[obj_list[0]]) for case in case_ids]
            with open("df_idf.p", "rb") as f:
                df_idf = pickle.load(f).append(pd.DataFrame({'total iterations[IDF]': [iters],
                                                             'total time[IDF]': [total_time],
                                                             'final_objective[IDF]': z[-1]}))
            with open("df_idf.p", "wb") as f:
                pickle.dump(df_idf, f)

        elif prob.driver.options['tol'] == 1e-3:
            iters = len(CaseReader('cases_idf.sql').get_cases())
            cr = CaseReader('cases_idf.sql')
            case_ids = cr.get_cases()
            obj_list = ['obj']
            z = []
            a = ["str_count.p", "aer_count.p", "pro_count.p", "obj_count.p"]
            k = []
            for i in a:
                with open(i, "rb") as f:
                    k.append(pickle.load(f))
            [z.append(case.get_objectives(case)[obj_list[0]]) for case in case_ids]
            with open("df_idf_tol.p", "rb") as f:
                df_idf = pickle.load(f).append(pd.DataFrame({'11.total iterations[IDF_tol]': [iters],
                                                            '12.total time[IDF_tol]': [total_time],
                                                             '13.final_objective[IDF_tol]': z[-1],
                                                             '14.str_count[IDF_tol]': k[0],
                                                             '15.aer_count[IDF_tol]': k[1],
                                                             '16.pro_count[IDF_tol]': k[2],
                                                             '17.obj_count[IDF_tol]': k[3]
                                                             }))
            with open("df_idf_tol.p", "wb") as f:
                pickle.dump(df_idf, f)

        # cr = CaseReader('cases_idf.sql')
        # case_ids = cr.get_cases()
        # total = []
        # constraints_list = ['con_g1', 'con_g2', 'con_g2']
        # for k in range(3):
        #     for i in range(nx):
        #         z = []
        #         [z.append(case.get_constraints(case)[constraints_list[k] + str(i + 1)]) for case in case_ids]
        #         total.append(z)
        # # print(total)
        # b = np.array(total).reshape(nx * 3, len(case_ids))
        # pickle.dump(b, open("con_idf.p", "wb"))
        # fig = plt.figure(1)
        # ax = fig.add_subplot(111)
        # a = ax.pcolor(b, cmap='viridis')
        # plt.xlabel('IDF driver cases')
        # plt.ylabel('Constraints')
        # label = []
        # for i in range(3):
        #     for j in range(nx):
        #         label.append('g' + str(i + 1) + '_' + str(j + 1))
        # ax.set_yticklabels(label)
        # plt.yticks(np.arange(0, 3 * nx, 1))
        # fig.colorbar(a)
        #
        # # objective function history for IDF`
        # cr = CaseReader('cases_idf.sql')
        # case_ids = cr.get_cases()
        # obj_list = ['obj']
        # z = []
        # [z.append(case.get_objectives(case)[obj_list[0]]) for case in case_ids]
        # # print(z)
        # if prob.driver.options['tol'] == 1e-2:
        #     pickle.dump(z, open("obj_idf_tol.p", "wb"))
        #     fig = plt.figure(2)
        #     ax = fig.add_subplot(111)
        #     a = ax.plot(z)
        #     plt.xlabel('IDF driver cases')
        #     plt.ylabel('objective')
        # else:
        #     pickle.dump(z, open("obj_idf.p", "wb"))
        #     fig = plt.figure(2)
        #     ax = fig.add_subplot(111)
        #     a = ax.plot(z)
        #     plt.xlabel('IDF driver cases')
        #     plt.ylabel('objective')
        #
        # # norm of the residuals
        # cr = CaseReader('cases_idf.sql')
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
        # pickle.dump(d_sum, open("l2_norm_xdes_idf.p", "wb"))
        # fig = plt.figure(3)
        # ax = fig.add_subplot(111)
        # a = ax.plot(d_sum)
        # plt.xlabel('driver cases')
        # plt.ylabel('d_sum')
