import pandas as pd
import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
import pickle

from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder
from openmdao.test_suite.components.sellar_feature import SellarMDA
from openmdao.recorders.case_reader import CaseReader


df_mdf = pickle.load(open("df_mdf.p", "rb"))
df_idf = pickle.load(open("df_idf.p", "rb"))
df_idf_tol = pickle.load(open("df_idf_tol.p", "rb"))
# df_idf = df_idf.reset_index()
# df_mdf = df_mdf.reset_index()
# df_idf_tol = df_idf_tol.reset_index()
#
# df_idf = df_idf[df_idf.index != 7]
# df_mdf = df_mdf.drop(columns=['index'])
# df_idf = df_idf.drop(columns=['index'])
# df_idf_tol = df_idf_tol.drop(columns=['index'])
# df_idf = df_idf.reset_index()
# df_idf = df_idf.drop(columns=['index'])


print(len(df_mdf.index))
print(len(df_idf_tol.index))
# print (df_mdf)
df_idf_tol = df_idf_tol[:-1]
# df_idf_tol = df_idf_tol[df_idf_tol["15.pro_count[IDF_tol]"] != 1070 ]
# print (df_idf_tol)
#df_idf_tol = df_idf_tol[df_idf_tol["total iterations[IDF_tol]"] != 94]
# print (df_idf_tol)
# df_mdf = df_mdf[df_mdf["total iterations[MDF]"] != 177]
# 1 print (df_mdf)

# pickle.dump(df_mdf, open("df_mdf.p", "wb"))
# pickle.dump(df_idf, open("df_idf.p", "wb"))
# pickle.dump(df_idf_tol, open("df_idf_tol.p", "wb"))
# result(variable_coupling_density) = pd.concat([df_mdf, df_idf, df_idf_tol], axis=1, sort='True')
# print(result(variable_coupling_density))
# result(variable_coupling_density).to_csv('result_coupling_density2.csv')