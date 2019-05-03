import pandas as pd
import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
import pickle

from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder
from openmdao.test_suite.components.sellar_feature import SellarMDA
from openmdao.recorders.case_reader import CaseReader

np.set_printoptions(threshold=np.inf)

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
print (df_mdf)
print(df_idf_tol)
# df_idf_tol = df_idf_tol[0:-1]
# df_idf = df_idf[df_idf["total iterations[IDF]"] != 147]
# print (df_idf)
# df_idf_tol = df_idf_tol[df_idf_tol["total iterations[IDF_tol]"] != 94]
# print (df_idf_tol)
# df_mdf = df_mdf[df_mdf["total iterations[MDF]"] != 177]
# 1 print (df_mdf)

# pickle.dump(df_mdf, open("df_mdf.p", "wb"))
# pickle.dump(df_idf, open("df_idf.p", "wb"))
# pickle.dump(df_idf_tol, open("df_idf_tol.p", "wb"))
# result = pd.concat([df_mdf, df_idf, df_idf_tol], axis=1, sort='True')
# print(result)
# result.to_csv('result_coupling_density2.csv')

result = pd.concat([df_mdf, df_idf_tol], axis=1)
result.to_csv('results(averaging out runs)/result(nx = 20)/results.csv')