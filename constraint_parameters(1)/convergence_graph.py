from __future__ import division, print_function
from matplotlib.pylab import plt
import numpy as np
import pickle
# from __future__ import division, print_function
from openmdao.api import Problem, IndepVarComp, ExplicitComponent
from openmdao.api import Group, Problem
from openmdao.api import IndepVarComp, ExecComp
from openmdao.api import NonlinearBlockGS, ScipyKrylov, LinearBlockGS, NewtonSolver, CaseReader
from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder
from openmdao.test_suite.components.sellar_feature import SellarMDA

nx = int(pickle.load(open("input_values.p", "rb"))[0])
con_idf = pickle.load(open("con_idf.p", "rb"))
con_mdf = pickle.load(open("con_mdf.p", "rb"))
obj_mdf = pickle.load(open("obj_mdf.p", "rb"))
obj_idf = pickle.load(open("obj_idf.p", "rb"))
obj_idf_tol = pickle.load(open("obj_idf_tol.p", "rb"))
fig = plt.figure(1)
ax = fig.add_subplot(211)
a = ax.pcolor(con_mdf, cmap='viridis')
print(con_mdf)
plt.xlabel('MDF driver cases', fontsize=12)
plt.xticks(np.arange(0, len(con_mdf), 1))
plt.ylabel('Constraints', fontsize=12)
label = []
for i in range(3):
    for j in range(nx):
        if j == 0:
            label.append('g' + '_' + str(i + 1) + '_' + str(j + 1))
        else:
            label.append(str(j + 1))
ax.set_yticklabels(label, fontdict={'fontsize': 7})
plt.yticks(np.arange(0, 3 * nx, 1))
fig.colorbar(a)

ax = fig.add_subplot(212)
a = ax.pcolor(con_idf, cmap='viridis')
plt.xlabel('IDF driver cases', fontsize=12)
plt.ylabel('Constraints', fontsize=12)
label = []
for i in range(3):
    for j in range(nx):
        if j == 0:
            label.append('g' + '_' + str(i + 1) + '_' + str(j + 1))
        else:
            label.append(str(j + 1))
ax.set_yticklabels(label, fontdict={'fontsize': 7})
plt.yticks(np.arange(0, 3 * nx, 1))
fig.colorbar(a)
fig.tight_layout()

fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
a = ax.plot(obj_mdf, label='MDF')
b = ax.plot(obj_idf, label='IDF')
#c = ax.plot(obj_idf_tol, label='IDF_TOL = 1e-3')
plt.xlabel('driver cases', fontsize=12)
plt.gca().legend(('MDF', 'IDF_TOL', 'IDF_TOL = 1e-3'))
plt.ylabel('objective', fontsize=12)
fig2.tight_layout()
plt.show()
