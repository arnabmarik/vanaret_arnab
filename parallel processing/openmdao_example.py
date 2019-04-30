import numpy as np
from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
from openmdao.test_suite.components.sellar_feature import SellarMDA

prob = Problem(model=SellarMDA())

model = prob.model
model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                          upper=np.array([10.0, 10.0]))
model.add_design_var('x', lower=0.0, upper=10.0)
model.add_objective('obj')
model.add_constraint('con1', upper=0.0)
model.add_constraint('con2', upper=0.0)

driver = prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9)

driver.add_recorder(SqliteRecorder('cases.sql'))

prob.setup()
prob.set_solver_print(0)
prob.run_driver()
print(driver.iter_count)
prob.cleanup()

# cr = CaseReader('cases.sql')
#
# case_ids = cr.list_cases()
#
# print(len(case_ids))
# print(case_ids)
# print('')
#
# for case_id in case_ids:
#     case = cr.get_case(case_id)
#     print(case)

d_mdf = result[result["ny"] == j]["total time[IDF_tol]"]
d_mdf = d_mdf.to_numpy()
a_mdf.append(np.mean(d_mdf))
ax.scatter((np.ones(np.size(d_mdf)) * (j - 2)).astype(int), d_mdf)