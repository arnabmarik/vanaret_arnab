import pandas as pd
import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
import pickle

from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder
from openmdao.test_suite.components.sellar_feature import SellarMDA
from openmdao.recorders.case_reader import CaseReader


"""/home/arnab/Downloads/thesis_project/github_clones/venv/bin/python /home/arnab/Downloads/thesis_project/github_clones/kadmos/examples/scripts/build_database_coupling_density.py
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 28 design variables and 22 response variables (objectives and nonlinear constraints).

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -7.41656621759
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 43 design variables and 37 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -8.03689749221
            Iterations: 17
            Function evaluations: 22
            Gradient evaluations: 17
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/pandas/core/frame.py:6692: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
of pandas will change to not sort by default.

To accept the future behavior, pass 'sort=False'.

To retain the current behavior and silence the warning, pass 'sort=True'.

  sort=sort)
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -8.03943423682
            Iterations: 8
            Function evaluations: 10
            Gradient evaluations: 8
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/matplotlib/cbook/deprecation.py:107: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
  warnings.warn(message, mplDeprecation, stacklevel=1)

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -8.03682478831
            Iterations: 10
            Function evaluations: 11
            Gradient evaluations: 10
Optimization Complete
-----------------------------------
z_opt= [0.42039373 0.41889174 0.51588159 1.         0.6734233  0.49676746
 0.44095506]
x1_opt= [1.         0.83715858 1.         0.37992508 1.         1.
 1.        ]
x2_opt= [0.93887906 0.97022755 0.80860103 0.16759069 0.         0.55724205
 0.        ]
x3_opt= [1.         1.         0.82691572 0.22352034 0.68264629 0.16667142
 1.        ]

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -10.0674142014
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 48 design variables and 42 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -10.6469687357
            Iterations: 21
            Function evaluations: 26
            Gradient evaluations: 21
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -10.6484494426
            Iterations: 9
            Function evaluations: 13
            Gradient evaluations: 9
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -10.6468913824
            Iterations: 15
            Function evaluations: 15
            Gradient evaluations: 15
Optimization Complete
-----------------------------------
z_opt= [0.42560984 0.8465421  0.41352809 0.64757894 0.04937101 0.72904902
 0.5879519 ]
x1_opt= [0.5        0.5        1.         0.57993518 1.         1.
 1.        ]
x2_opt= [0.         1.         0.69475177 0.75313417 0.50584264 0.
 0.78204821]
x3_opt= [0.32276121 0.45165588 0.44108467 0.26290963 1.         0.64118792
 1.        ]

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -12.6033903967
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 53 design variables and 47 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -12.6206544957
            Iterations: 25
            Function evaluations: 34
            Gradient evaluations: 25
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -12.625557148
            Iterations: 11
            Function evaluations: 16
            Gradient evaluations: 11
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -12.6204997718
            Iterations: 11
            Function evaluations: 11
            Gradient evaluations: 11
Optimization Complete
-----------------------------------
z_opt= [0.67256913 0.33203899 0.44882818 0.51102228 0.40372376 0.60057779
 0.32902805]
x1_opt= [4.99627408e-01 1.00000000e+00 1.00000000e+00 1.00000000e+00
 1.00000000e+00 4.13500629e-17 1.00000000e+00]
x2_opt= [0.6939385  0.2153675  0.         0.84471414 0.51222466 0.73977793
 0.74195381]
x3_opt= [0.54409771 0.5732515  0.36256922 0.4960954  0.28582875 0.2626663
 0.20859595]

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -15.0415150554
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 58 design variables and 52 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -15.3440582006
            Iterations: 22
            Function evaluations: 31
            Gradient evaluations: 22
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -15.3468116095
            Iterations: 13
            Function evaluations: 19
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -15.3438336071
            Iterations: 13
            Function evaluations: 14
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
z_opt= [0.41046501 0.7735561  0.75186903 0.51619336 0.52969785 0.35494308
 0.14239845]
x1_opt= [1.         1.         1.         1.         1.         0.24739042
 0.91160975]
x2_opt= [0.15987308 0.17463606 1.         0.22276869 0.6411846  0.32525882
 0.410838  ]
x3_opt= [0.1482216  1.         0.30519872 0.79773587 0.62922421 0.02963071
 0.57060101]

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -17.4592977207
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 63 design variables and 57 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -17.2915845905
            Iterations: 25
            Function evaluations: 39
            Gradient evaluations: 25
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -17.3087742977
            Iterations: 14
            Function evaluations: 26
            Gradient evaluations: 14
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -17.2912984186
            Iterations: 10
            Function evaluations: 11
            Gradient evaluations: 10
Optimization Complete
-----------------------------------
z_opt= [0.43369186 0.32707936 0.8445393  0.38301402 0.46667631 0.6347805
 0.93086926]
x1_opt= [0.89069465 0.86656779 0.89360727 0.95200031 1.         1.
 0.24595222]
x2_opt= [0.3386518  0.39343064 0.27721062 0.36853492 0.57154413 0.81980783
 0.48023376]
x3_opt= [0.49749025 0.19283833 0.54963594 0.4760434  0.30410714 0.22500409
 0.1659734 ]

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -20.1893191379
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 68 design variables and 62 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -19.8308936637
            Iterations: 30
            Function evaluations: 50
            Gradient evaluations: 30
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -19.8545402784
            Iterations: 11
            Function evaluations: 20
            Gradient evaluations: 11
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -19.8305991287
            Iterations: 20
            Function evaluations: 28
            Gradient evaluations: 20
Optimization Complete
-----------------------------------
z_opt= [0.71994637 0.27671685 0.29534743 0.12343739 0.53019171 0.29596462
 0.94397141]
x1_opt= [0.858648   0.93208921 0.92221749 0.99999981 1.         0.51359599
 0.96039279]
x2_opt= [0.14239527 0.25565936 0.72273279 0.23022234 0.43186346 0.3623674
 0.27579958]
x3_opt= [0.01387356 0.21174122 0.17160267 1.         0.54612842 0.90749435
 1.        ]

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -22.7973288477
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 73 design variables and 67 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -22.7012639542
            Iterations: 26
            Function evaluations: 38
            Gradient evaluations: 26
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -22.7030158673
            Iterations: 19
            Function evaluations: 31
            Gradient evaluations: 19
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -23.0761048643
            Iterations: 13
            Function evaluations: 14
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
z_opt= [0.5062447  1.         0.         0.41808014 0.86957463 0.70451383
 0.69652326]
x1_opt= [1.         1.         0.20236662 1.         1.         0.30522977
 1.        ]
x2_opt= [0.         0.         0.44525036 0.22642051 0.35756177 0.43860917
 0.66769868]
x3_opt= [0.         0.4895343  0.64496909 1.         0.24280459 0.68691182
 0.57044184]

===
Mda
===
NL: NLBGS Converged in 14 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -25.2687117228
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 78 design variables and 72 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -24.1671624044
            Iterations: 28
            Function evaluations: 52
            Gradient evaluations: 28
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -24.1698038197
            Iterations: 21
            Function evaluations: 37
            Gradient evaluations: 21
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 14 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 14 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -24.1664093901
            Iterations: 18
            Function evaluations: 34
            Gradient evaluations: 18
Optimization Complete
-----------------------------------
z_opt= [0.62031498 0.22786233 0.16425196 0.48455184 0.36368986 0.4782881
 1.        ]
x1_opt= [0.12933774 1.         0.30381325 1.         1.         1.
 0.41050971]
x2_opt= [0.61161816 0.11102459 0.25085497 0.92258591 0.58942152 0.2955412
 0.60520914]
x3_opt= [0.02173883 1.         0.27161647 0.18380522 0.10693441 0.53362294
 0.18813653]

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -7.52735405171
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.83195035947
            Iterations: 46
            Function evaluations: 73
            Gradient evaluations: 46
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.82605788579
            Iterations: 8
            Function evaluations: 9
            Gradient evaluations: 8
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.83190206618
            Iterations: 25
            Function evaluations: 26
            Gradient evaluations: 25
Optimization Complete
-----------------------------------
z_opt= [0.71383482 0.40385191 0.39328978 0.42032081 0.34051601 0.82542855
 0.62844271]
x1_opt= [1.         1.         0.33657751 1.         1.         0.3648376
 1.        ]
x2_opt= [5.73213497e-01 4.25051611e-01 9.05999179e-01 2.31229185e-01
 7.84836648e-01 1.11435363e-16 2.21474153e-01]
x3_opt= [0.52634451 1.         0.16303719 0.52945218 0.3877386  0.91334378
 0.40257558]

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -10.0912385395
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -10.0655331217
            Iterations: 33
            Function evaluations: 45
            Gradient evaluations: 33
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -10.0637014735
            Iterations: 12
            Function evaluations: 14
            Gradient evaluations: 12
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -10.0653451423
            Iterations: 17
            Function evaluations: 18
            Gradient evaluations: 17
Optimization Complete
-----------------------------------
z_opt= [0.50896922 0.51643049 0.60969997 0.47382603 0.76045503 0.33236549
 0.41028964]
x1_opt= [3.47313348e-01 4.49798520e-16 5.15807340e-01 9.19446830e-01
 1.00000000e+00 9.33951863e-01 5.60274412e-01]
x2_opt= [0.94845592 0.2106536  0.358537   0.83082309 0.3683905  0.36706445
 0.        ]
x3_opt= [0.62946093 0.05840447 0.38889638 0.46490816 0.33811048 0.48496987
 0.54738922]

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -12.4712213629
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -12.2649730191
            Iterations: 24
            Function evaluations: 34
            Gradient evaluations: 24
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -12.2709346806
            Iterations: 12
            Function evaluations: 19
            Gradient evaluations: 12
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -12.141551858
            Iterations: 15
            Function evaluations: 16
            Gradient evaluations: 15
Optimization Complete
-----------------------------------
z_opt= [0.44222167 0.46948537 0.45597523 0.21488746 0.70244092 0.59638341
 0.45999478]
x1_opt= [0.943198   0.71622249 0.98710036 0.88600322 0.2523705  1.
 0.5       ]
x2_opt= [0.28372295 0.88470058 0.5460669  0.2115925  0.7383899  0.58965835
 1.        ]
x3_opt= [1.         0.04175257 0.5389977  0.21322848 0.203773   0.54614209
 0.32713986]

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -14.99017394
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -14.5665426762
            Iterations: 34
            Function evaluations: 59
            Gradient evaluations: 34
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -14.5672587326
            Iterations: 20
            Function evaluations: 34
            Gradient evaluations: 20
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -14.5661664653
            Iterations: 30
            Function evaluations: 60
            Gradient evaluations: 30
Optimization Complete
-----------------------------------
z_opt= [0.4403991  0.60149903 0.45425501 0.63613076 0.39704416 0.6799295
 0.68003237]
x1_opt= [0.57008297 1.         0.51499065 1.         0.37720528 1.
 1.        ]
x2_opt= [5.48573593e-01 2.51671017e-01 8.29460376e-02 1.07589699e-15
 9.87371494e-01 3.83214614e-02 8.55765043e-01]
x3_opt= [2.17390248e-02 1.44260836e-01 1.10174573e-01 2.17292610e-02
 3.43357230e-01 4.37294730e-01 1.36653210e-15]

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -17.5198855266
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.9543206685
            Iterations: 29
            Function evaluations: 60
            Gradient evaluations: 29
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.973938124
            Iterations: 11
            Function evaluations: 19
            Gradient evaluations: 11
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.9539083622
            Iterations: 13
            Function evaluations: 16
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
z_opt= [0.61266844 0.51112613 0.52831903 1.         0.23922247 0.60799141
 0.58708284]
x1_opt= [0.33685501 1.         0.99993977 0.13042634 0.52272745 0.34760649
 1.        ]
x2_opt= [0.46284783 0.9032593  1.         0.09356858 1.         1.
 0.39326464]
x3_opt= [0.09916931 0.         0.11730799 1.         0.03872858 0.
 0.12866686]

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -20.0700813706
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -19.2835076217
            Iterations: 23
            Function evaluations: 46
            Gradient evaluations: 23
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -19.2905149414
            Iterations: 14
            Function evaluations: 22
            Gradient evaluations: 14
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -19.2899932295
            Iterations: 15
            Function evaluations: 15
            Gradient evaluations: 15
Optimization Complete
-----------------------------------
z_opt= [0.65962856 0.16107225 0.53767977 0.1405946  0.25331914 0.41234444
 0.78837087]
x1_opt= [0.46141145 0.52880294 1.         0.20205491 1.         1.
 0.34473738]
x2_opt= [0.18046031 0.22886125 1.         0.39904321 1.         0.49548019
 0.80244107]
x3_opt= [0.2697501  1.         0.26438176 1.         0.25263917 1.
 0.06848825]

===
Mda
===
NL: NLBGS Converged in 15 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -22.5209050618
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -21.1932233341
            Iterations: 33
            Function evaluations: 54
            Gradient evaluations: 33
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -21.1960209638
            Iterations: 23
            Function evaluations: 39
            Gradient evaluations: 23
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 15 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -20.6545160087
            Iterations: 17
            Function evaluations: 23
            Gradient evaluations: 17
Optimization Complete
-----------------------------------
z_opt= [0.53398174 0.08578072 0.40193299 0.65616177 0.30813931 0.27848461
 1.        ]
x1_opt= [1.         1.         0.16300374 0.09224959 1.         1.
 1.        ]
x2_opt= [0.19293792 0.20125907 0.2820736  1.         0.28330892 1.
 0.75434123]
x3_opt= [0.12742041 0.         0.15073655 0.17754086 0.2254596  0.3827513
 0.23868691]

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -25.4103933396
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -24.225937694
            Iterations: 48
            Function evaluations: 79
            Gradient evaluations: 48
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -24.2545177639
            Iterations: 21
            Function evaluations: 41
            Gradient evaluations: 21
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -24.2251894807
            Iterations: 16
            Function evaluations: 18
            Gradient evaluations: 16
Optimization Complete
-----------------------------------
z_opt= [0.39096184 0.60663351 0.39976707 0.11878506 0.42528135 0.45555428
 0.13689105]
x1_opt= [0.30034844 1.         1.         1.         1.         1.
 1.        ]
x2_opt= [0.82284567 1.         0.33298398 1.         0.98180167 0.08979091
 0.12061151]
x3_opt= [2.98570897e-01 6.28717643e-01 9.22748663e-01 6.65518731e-01
 4.41540341e-01 2.57497973e-14 1.50215177e-01]

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -7.54377367547
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.53838668876
            Iterations: 19
            Function evaluations: 23
            Gradient evaluations: 19
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.54094646573
            Iterations: 9
            Function evaluations: 12
            Gradient evaluations: 9
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.53831448459
            Iterations: 15
            Function evaluations: 15
            Gradient evaluations: 15
Optimization Complete
-----------------------------------
z_opt= [0.57564361 0.28455462 0.51360222 0.52965644 0.55044996 0.69235767
 0.68766205]
x1_opt= [9.27818739e-01 5.21304677e-01 3.09855385e-01 4.09963636e-01
 1.73663322e-16 1.00000000e+00 5.83991976e-01]
x2_opt= [5.72633392e-01 2.57445005e-01 5.41424153e-01 3.71395875e-16
 1.00000000e+00 3.22928107e-01 3.38943719e-01]
x3_opt= [0.7332007  0.39056604 0.42922319 0.3008413  0.52928608 0.41779131
 0.19351419]

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -9.96681117352
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.7508416405
            Iterations: 17
            Function evaluations: 23
            Gradient evaluations: 17
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.75121873089
            Iterations: 10
            Function evaluations: 11
            Gradient evaluations: 10
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.75068316082
            Iterations: 22
            Function evaluations: 30
            Gradient evaluations: 22
Optimization Complete
-----------------------------------
z_opt= [0.20304486 0.38463114 0.40291369 0.71363831 0.3888132  0.85869475
 0.19167426]
x1_opt= [0.2800137  0.54437881 1.         1.         0.51659291 0.
 0.41244036]
x2_opt= [0.14102474 0.76061662 1.         0.55502428 0.         0.25380536
 0.82431904]
x3_opt= [0.35247022 0.48067023 0.78999086 1.         1.         0.33410118
 0.42970233]

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -12.6112041032
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -12.5217481141
            Iterations: 23
            Function evaluations: 31
            Gradient evaluations: 23
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -12.5339507024
            Iterations: 8
            Function evaluations: 11
            Gradient evaluations: 8
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -12.5215997848
            Iterations: 11
            Function evaluations: 12
            Gradient evaluations: 11
Optimization Complete
-----------------------------------
z_opt= [0.41691911 0.73589198 0.23307611 0.4780126  0.60841419 0.32107432
 0.49597231]
x1_opt= [0.83993549 0.90965018 0.37862395 0.5834185  0.5950806  0.50004284
 0.9416023 ]
x2_opt= [0.86536212 0.42881227 0.83518085 0.99092068 0.17758355 0.15561291
 0.22647876]
x3_opt= [0.95614194 0.4189823  0.46789035 0.34268473 0.46219897 0.36363215
 0.52162142]

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -15.0426406856
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -15.1213047223
            Iterations: 20
            Function evaluations: 27
            Gradient evaluations: 20
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -15.1293168067
            Iterations: 9
            Function evaluations: 13
            Gradient evaluations: 9
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -15.1211727117
            Iterations: 12
            Function evaluations: 13
            Gradient evaluations: 12
Optimization Complete
-----------------------------------
z_opt= [0.4705589  0.28475078 0.45550934 0.74322868 0.38158541 0.5081554
 0.60565209]
x1_opt= [0.44390771 0.         1.         1.         1.         0.61653996
 1.        ]
x2_opt= [1.         0.5569966  0.49208538 0.30539638 0.21914818 1.
 0.33277043]
x3_opt= [0.57749999 0.15210868 0.         1.         0.3940444  0.47226426
 0.57489743]

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -17.6094503544
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.7025916393
            Iterations: 20
            Function evaluations: 27
            Gradient evaluations: 20
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.7047286106
            Iterations: 13
            Function evaluations: 19
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.7023318386
            Iterations: 9
            Function evaluations: 10
            Gradient evaluations: 9
Optimization Complete
-----------------------------------
z_opt= [0.29214755 0.37837615 0.26064835 0.71017747 0.51254169 0.42457203
 0.41405787]
x1_opt= [1.         0.35425761 0.42434399 0.43123024 0.38215055 0.41716688
 0.34408701]
x2_opt= [0.14875237 0.72529852 0.93293489 0.98995276 1.         0.73533098
 0.22528493]
x3_opt= [0.63808487 0.25390586 0.45968761 0.22277669 0.51096745 0.28350094
 0.38837429]

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -20.0481525805
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -19.3736331646
            Iterations: 21
            Function evaluations: 27
            Gradient evaluations: 21
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -19.3754674864
            Iterations: 13
            Function evaluations: 19
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -19.3734313569
            Iterations: 13
            Function evaluations: 14
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
z_opt= [0.47648498 0.38970986 0.44147449 1.         0.32630921 0.53957988
 0.52183922]
x1_opt= [0.21602406 0.52719819 0.52693089 0.37279449 0.29888655 0.51999341
 0.42506344]
x2_opt= [0.33338787 0.06709779 0.85540988 0.57109946 0.35671436 0.63126536
 0.62855997]
x3_opt= [1.00000000e+00 3.98263853e-01 2.76065690e-01 4.34831417e-01
 1.68751726e-01 6.34292997e-16 5.00681875e-01]

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -22.7931180576
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -21.1892418071
            Iterations: 35
            Function evaluations: 58
            Gradient evaluations: 35
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -21.2176509803
            Iterations: 18
            Function evaluations: 33
            Gradient evaluations: 18
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -21.188733034
            Iterations: 14
            Function evaluations: 15
            Gradient evaluations: 14
Optimization Complete
-----------------------------------
z_opt= [0.56133414 0.28612537 0.41765965 0.24126758 0.4368963  0.23684295
 0.5375496 ]
x1_opt= [1.         0.35052183 1.         1.         0.25396511 0.34396816
 1.        ]
x2_opt= [1.         0.23696386 0.98396473 0.03872184 0.24034656 1.
 1.        ]
x3_opt= [0.32256364 0.30156819 0.20692718 0.12684937 0.59018156 0.32774172
 0.6925085 ]

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -25.0904427018
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -22.4024354246
            Iterations: 70
            Function evaluations: 125
            Gradient evaluations: 70
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -22.4181656247
            Iterations: 33
            Function evaluations: 70
            Gradient evaluations: 33
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -22.3007657449
            Iterations: 42
            Function evaluations: 113
            Gradient evaluations: 42
Optimization Complete
-----------------------------------
z_opt= [0.49255517 0.41981032 0.16446385 0.62332696 0.42314849 0.24605942
 0.03470049]
x1_opt= [1.         0.07343973 1.         0.12670285 1.         1.
 1.        ]
x2_opt= [2.68184947e-01 1.00000000e+00 1.00000000e+00 2.56801352e-12
 1.00000000e+00 1.00000000e+00 1.00000000e+00]
x3_opt= [0.0217386  0.2514338  0.02173881 0.29445559 0.1644048  0.32080899
 0.26287642]

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -7.38350058667
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.59132047859
            Iterations: 27
            Function evaluations: 39
            Gradient evaluations: 27
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.5937415454
            Iterations: 11
            Function evaluations: 15
            Gradient evaluations: 11
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.59127085543
            Iterations: 26
            Function evaluations: 44
            Gradient evaluations: 26
Optimization Complete
-----------------------------------
z_opt= [0.35063951 0.45787614 0.51466671 0.72531319 0.50870032 0.40841911
 0.77213741]
x1_opt= [0.65981235 0.72425256 1.         0.         1.         1.
 0.        ]
x2_opt= [0.09713    0.79715372 0.         0.65718337 0.36761693 1.
 0.56115442]
x3_opt= [0.51748618 0.02173492 0.35646342 0.4860224  0.67868076 0.43016082
 0.3391333 ]

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -9.91864658792
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.53847670722
            Iterations: 21
            Function evaluations: 29
            Gradient evaluations: 21
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.53949760851
            Iterations: 14
            Function evaluations: 22
            Gradient evaluations: 14
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.53834103573
            Iterations: 12
            Function evaluations: 12
            Gradient evaluations: 12
Optimization Complete
-----------------------------------
z_opt= [0.35436662 0.41609178 0.42587876 0.4687818  0.27410014 0.35302208
 0.68274518]
x1_opt= [0.56972848 0.30325901 0.50203038 0.50305874 1.         0.776216
 0.56648061]
x2_opt= [2.59119217e-01 9.04614107e-01 8.31245969e-16 5.10690288e-01
 8.18644726e-01 7.62073942e-01 7.27706130e-01]
x3_opt= [0.1564393  0.60974349 0.41194081 0.35318659 0.56701759 0.0336612
 0.66329683]

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -12.3688594933
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -11.7499697269
            Iterations: 26
            Function evaluations: 40
            Gradient evaluations: 26
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -11.7573994498
            Iterations: 12
            Function evaluations: 19
            Gradient evaluations: 12
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -11.7497057572
            Iterations: 13
            Function evaluations: 14
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
z_opt= [0.47934521 0.34717149 0.42882881 0.53621678 0.58481686 0.65299499
 0.59161795]
x1_opt= [1.         0.26304615 0.32911423 0.33130659 1.         0.34454193
 0.34094809]
x2_opt= [2.57847533e-15 9.00695687e-01 1.00000000e+00 5.15315246e-02
 8.47216691e-01 9.20395844e-01 7.71447738e-02]
x3_opt= [0.23770524 0.16332639 0.40728239 0.28961943 0.33987491 0.31242204
 0.51057355]

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -14.8946423745
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Iteration limit exceeded    (Exit mode 9)
            Current function value: -13.135444271
            Iterations: 101
            Function evaluations: 526
            Gradient evaluations: 98
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Iteration limit exceeded    (Exit mode 9)
            Current function value: -13.135444271
            Iterations: 101
            Function evaluations: 526
            Gradient evaluations: 98
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Positive directional derivative for linesearch    (Exit mode 8)
            Current function value: -12.9521777509
            Iterations: 97
            Function evaluations: 474
            Gradient evaluations: 93
Optimization FAILED.
Positive directional derivative for linesearch
-----------------------------------
z_opt= [0.318372   0.91087452 0.38368289 0.4238514  0.20689328 0.06616018
 0.47765985]
x1_opt= [6.16970695e-01 7.80694711e-01 1.84031697e-21 9.13183899e-01
 1.88403582e-21 3.63621513e-01 1.41507827e-21]
x2_opt= [0.         1.         1.         1.         1.         1.
 0.94309781]
x3_opt= [3.78440068e-01 6.57763534e-01 3.47676824e-21 4.13259075e-01
 3.99722193e-01 3.93185496e-01 7.37540649e-01]

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -17.4957137595
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -17.0427313749
            Iterations: 27
            Function evaluations: 44
            Gradient evaluations: 27
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -17.0549259348
            Iterations: 11
            Function evaluations: 17
            Gradient evaluations: 11
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -17.042533583
            Iterations: 21
            Function evaluations: 32
            Gradient evaluations: 21
Optimization Complete
-----------------------------------
z_opt= [0.53901841 0.46603806 0.4390637  0.49258279 0.2860759  0.45547684
 0.32648275]
x1_opt= [1.         0.50895934 0.38382263 1.         0.62566704 1.
 1.        ]
x2_opt= [0.34522261 0.60363048 0.98668214 0.21106782 0.67821722 0.43826791
 1.        ]
x3_opt= [0.1145098  0.88338006 0.31512017 0.23886494 0.13763095 0.16381965
 0.40675944]

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -20.1507057743
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -18.1587164527
            Iterations: 49
            Function evaluations: 78
            Gradient evaluations: 49
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -18.1621087399
            Iterations: 13
            Function evaluations: 20
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -17.8510188473
            Iterations: 15
            Function evaluations: 18
            Gradient evaluations: 15
Optimization Complete
-----------------------------------
z_opt= [0.42301382 0.47375723 0.16757741 0.50002345 0.2029719  0.17204877
 0.2605425 ]
x1_opt= [0.23260055 1.         0.22362445 0.16351702 1.         1.
 0.1512013 ]
x2_opt= [1.00000000e+00 1.00000000e+00 1.00000000e+00 4.21409086e-01
 3.19169710e-12 1.00000000e+00 1.00000000e+00]
x3_opt= [0.88637072 0.04459758 0.39090455 0.08053732 0.44942643 0.44371211
 0.77989914]

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -22.6783002616
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -21.2959503038
            Iterations: 38
            Function evaluations: 70
            Gradient evaluations: 38
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -21.3180092117
            Iterations: 19
            Function evaluations: 39
            Gradient evaluations: 19
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -21.3051340435
            Iterations: 15
            Function evaluations: 16
            Gradient evaluations: 15
Optimization Complete
-----------------------------------
z_opt= [0.38275203 0.3076188  0.80735943 0.27136248 0.38305343 0.2933764
 0.64705273]
x1_opt= [1.         1.         1.         0.39012555 1.         0.39715097
 0.32427159]
x2_opt= [0.17293454 0.95216237 0.95544786 0.12161308 0.99229033 0.27783812
 0.79356883]
x3_opt= [3.00121751e-01 2.11723352e-01 3.18139256e-01 2.38624049e-01
 2.25955419e-01 5.02973336e-16 3.36139754e-01]

===
Mda
===
NL: NLBGS Converged in 14 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -25.4362501828
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -22.9508742435
            Iterations: 55
            Function evaluations: 105
            Gradient evaluations: 55
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -22.9642364834
            Iterations: 33
            Function evaluations: 70
            Gradient evaluations: 33
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 14 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -23.0256145824
            Iterations: 25
            Function evaluations: 49
            Gradient evaluations: 25
Optimization Complete
-----------------------------------
z_opt= [0.38430205 0.32328016 0.49045551 0.40166069 0.16208283 0.27363652
 0.46895876]
x1_opt= [0.57287646 0.4923145  0.99999964 0.94153678 0.34171256 0.56977661
 0.53528132]
x2_opt= [9.18293111e-01 9.37918090e-01 1.00000000e+00 9.98690001e-01
 5.68679651e-15 9.51636481e-01 0.00000000e+00]
x3_opt= [0.62333736 0.35112702 0.13700701 0.0839767  0.53571665 0.32265116
 0.02173913]

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -7.46768311482
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.60465710972
            Iterations: 18
            Function evaluations: 22
            Gradient evaluations: 18
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.60983771526
            Iterations: 7
            Function evaluations: 9
            Gradient evaluations: 7
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.6045981147
            Iterations: 14
            Function evaluations: 14
            Gradient evaluations: 14
Optimization Complete
-----------------------------------
z_opt= [0.47625951 0.46039144 0.61462964 0.42836146 0.51420609 0.48660522
 0.63595774]
x1_opt= [0.47665968 0.36793702 1.         1.         0.47091959 0.47091959
 1.        ]
x2_opt= [6.78155376e-01 7.32437646e-01 9.50619794e-16 9.33131843e-01
 8.38321742e-16 4.24574641e-01 8.07918151e-01]
x3_opt= [8.35616533e-02 5.39871437e-16 3.65888125e-01 3.91534773e-01
 9.72656228e-01 3.21182941e-01 5.02596154e-01]

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -9.9699375573
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.63510630947
            Iterations: 28
            Function evaluations: 34
            Gradient evaluations: 28
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.63358577673
            Iterations: 11
            Function evaluations: 14
            Gradient evaluations: 11
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.63499123621
            Iterations: 22
            Function evaluations: 27
            Gradient evaluations: 22
Optimization Complete
-----------------------------------
z_opt= [0.53745815 0.38269142 0.24662038 0.37329304 0.42134052 0.65158701
 0.59077342]
x1_opt= [0.58923141 0.58324386 0.51957394 0.57986549 0.58572821 1.
 1.        ]
x2_opt= [0.39070888 0.15078507 0.         0.31754989 0.93606209 0.94282377
 0.99253616]
x3_opt= [0.43990154 0.32321205 0.39774581 0.15252606 0.34429186 0.37950327
 0.36249227]

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -12.3893238193
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -11.9814964147
            Iterations: 20
            Function evaluations: 29
            Gradient evaluations: 20
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -11.9952557899
            Iterations: 9
            Function evaluations: 15
            Gradient evaluations: 9
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -11.9791343459
            Iterations: 13
            Function evaluations: 13
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
z_opt= [1.         0.12166334 0.43240836 0.43854871 0.48878623 0.58792379
 0.32002527]
x1_opt= [0.42570492 0.35849684 0.47666602 0.42424161 0.39672854 1.
 0.39672854]
x2_opt= [1.00000000e+00 1.67479147e-01 9.04690222e-01 4.86442735e-01
 9.17866247e-01 9.86305647e-01 1.57857177e-15]
x3_opt= [0.64819103 0.62803738 0.36237207 0.53107784 0.5606337  0.12671911
 0.17988309]

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -15.0484694563
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -14.6650783522
            Iterations: 28
            Function evaluations: 53
            Gradient evaluations: 28
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -14.6871613252
            Iterations: 9
            Function evaluations: 14
            Gradient evaluations: 9
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -14.6652890957
            Iterations: 21
            Function evaluations: 29
            Gradient evaluations: 21
Optimization Complete
-----------------------------------
z_opt= [0.24767346 0.43860484 0.81346981 0.50427955 0.66022702 0.34425276
 0.60539538]
x1_opt= [0.38640798 1.         0.43084841 0.33937286 0.43852872 1.
 1.        ]
x2_opt= [4.98720928e-01 0.00000000e+00 8.60952464e-01 3.66889383e-01
 8.74307198e-01 1.28224386e-14 1.00000000e+00]
x3_opt= [0.11975319 0.48054525 0.19678307 0.63331564 0.13986818 0.41592741
 0.2184109 ]

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -17.6890585559
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.1105546806
            Iterations: 30
            Function evaluations: 51
            Gradient evaluations: 30
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.1161004274
            Iterations: 17
            Function evaluations: 30
            Gradient evaluations: 17
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.0353191082
            Iterations: 26
            Function evaluations: 28
            Gradient evaluations: 26
Optimization Complete
-----------------------------------
z_opt= [0.8445439  0.03433012 0.70828252 0.65235043 0.04039087 0.26295473
 0.33865795]
x1_opt= [0.82021798 0.41202998 0.37409193 0.36452862 0.42647945 0.48259174
 0.5206165 ]
x2_opt= [2.29230253e-01 8.21357119e-01 1.00000000e+00 9.25151496e-01
 1.00000000e+00 6.20572720e-12 9.65014388e-01]
x3_opt= [4.95639714e-01 5.03576563e-01 5.04871767e-01 4.62011564e-01
 4.19954615e-01 1.00000000e+00 2.82441530e-12]

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -20.0567569884
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -18.3736355274
            Iterations: 44
            Function evaluations: 79
            Gradient evaluations: 44
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -18.3747091488
            Iterations: 27
            Function evaluations: 53
            Gradient evaluations: 27
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -18.3730675306
            Iterations: 55
            Function evaluations: 120
            Gradient evaluations: 54
Optimization Complete
-----------------------------------
z_opt= [0.14582743 0.40215841 0.36320898 0.32501399 0.70954444 0.61449968
 0.34964395]
x1_opt= [0.34289736 0.96580675 0.36964916 1.         0.40465862 0.41544908
 0.41428194]
x2_opt= [5.84330954e-01 4.31587283e-07 9.99999913e-01 1.00000000e+00
 9.06352247e-01 9.83486300e-01 9.61628525e-07]
x3_opt= [0.55502518 0.33232136 0.06067798 0.29566349 0.33902148 0.36705878
 0.54541211]

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -22.6789244025
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -20.4069373866
            Iterations: 39
            Function evaluations: 70
            Gradient evaluations: 39
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -20.4169444099
            Iterations: 25
            Function evaluations: 52
            Gradient evaluations: 25
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -20.4060108286
            Iterations: 21
            Function evaluations: 24
            Gradient evaluations: 21
Optimization Complete
-----------------------------------
z_opt= [0.68227753 0.23410624 0.58042787 0.65305263 0.15996275 0.19430698
 0.67651421]
x1_opt= [0.51705544 0.98085945 0.45067252 0.48110618 0.52188548 0.26412336
 0.47167787]
x2_opt= [0.60891635 0.14642317 0.04669366 1.         1.         0.97631891
 0.96110995]
x3_opt= [0.60513628 0.17030657 0.07823495 0.47370844 0.05474552 0.17234804
 0.1684658 ]

===
Mda
===
NL: NLBGS Converged in 14 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -25.187350228
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -23.5275065779
            Iterations: 31
            Function evaluations: 55
            Gradient evaluations: 31
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -23.54007026
            Iterations: 16
            Function evaluations: 31
            Gradient evaluations: 16
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 14 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -23.5270930074
            Iterations: 15
            Function evaluations: 17
            Gradient evaluations: 15
Optimization Complete
-----------------------------------
z_opt= [0.58549872 0.33512298 0.33775098 0.47188566 0.30689895 0.29652996
 0.30397529]
x1_opt= [0.5600636  0.55261085 0.53809049 0.5422211  0.58106026 0.59573585
 1.        ]
x2_opt= [0.06834389 1.         0.07387929 0.99671572 0.92412388 0.6193142
 0.94136745]
x3_opt= [0.37610852 0.41987941 0.76779052 0.23599207 0.22081493 0.19102222
 0.16687921]

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -7.46796893939
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.22347349088
            Iterations: 23
            Function evaluations: 28
            Gradient evaluations: 23
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.22315352033
            Iterations: 9
            Function evaluations: 9
            Gradient evaluations: 9
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.22338973592
            Iterations: 16
            Function evaluations: 17
            Gradient evaluations: 16
Optimization Complete
-----------------------------------
z_opt= [0.45071671 0.51769804 0.39994387 0.4610446  0.39700722 0.60303591
 0.37751908]
x1_opt= [0.57825344 0.57226751 0.22642735 0.42967415 0.55920837 0.54858962
 0.57226751]
x2_opt= [0.35022649 0.72471073 0.60877022 0.40839768 0.30646121 0.82284788
 0.86937278]
x3_opt= [0.84107114 0.3329241  0.47289463 0.60784422 0.34704995 0.16714908
 0.2346425 ]

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -10.0139079109
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.6785186664
            Iterations: 18
            Function evaluations: 23
            Gradient evaluations: 18
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.67953619175
            Iterations: 11
            Function evaluations: 15
            Gradient evaluations: 11
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.67842425305
            Iterations: 12
            Function evaluations: 13
            Gradient evaluations: 12
Optimization Complete
-----------------------------------
z_opt= [0.33654673 0.53354739 0.50340744 0.37589993 0.47061903 0.33426824
 0.62362729]
x1_opt= [0.4971892  0.49690439 0.4853019  1.         1.         0.48530191
 0.43275678]
x2_opt= [0.56349194 0.65097034 0.42254623 0.37348959 0.39903619 0.57561895
 0.87316471]
x3_opt= [0.49425556 0.61081786 0.27533267 0.247821   0.24204745 0.59327721
 0.23749942]

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -12.5097963572
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -11.2330722799
            Iterations: 25
            Function evaluations: 44
            Gradient evaluations: 25
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -11.2407257953
            Iterations: 14
            Function evaluations: 23
            Gradient evaluations: 14
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -11.208170835
            Iterations: 46
            Function evaluations: 80
            Gradient evaluations: 46
Optimization Complete
-----------------------------------
z_opt= [4.82914278e-01 5.04049996e-01 8.27324750e-11 8.41208213e-01
 6.81926039e-01 5.69924906e-01 6.98994493e-01]
x1_opt= [0.42224055 0.47898177 0.43986866 0.40562717 1.         0.30455435
 0.50064007]
x2_opt= [7.63294078e-11 7.62222087e-11 1.00000000e+00 1.00000000e+00
 1.00000000e+00 7.59904001e-11 8.64842960e-02]
x3_opt= [0.30978628 0.32230419 0.25123649 0.25552691 0.21686947 0.32930447
 0.34693389]

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -14.8703803992
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Iteration limit exceeded    (Exit mode 9)
            Current function value: -13.7898175058
            Iterations: 101
            Function evaluations: 419
            Gradient evaluations: 98
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Iteration limit exceeded    (Exit mode 9)
            Current function value: -13.7898175058
            Iterations: 101
            Function evaluations: 419
            Gradient evaluations: 98
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 2 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 3 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations

===
Mda
===
NL: NLBGS Converged in 4 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -13.5268658754
            Iterations: 101
            Function evaluations: 378
            Gradient evaluations: 98
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
z_opt= [0.55001948 0.36779397 0.35643105 0.36292879 0.62731249 0.49285291
 0.32330857]
x1_opt= [6.85132212e-01 6.68965034e-01 1.71519096e-11 6.85097946e-01
 9.13668164e-12 2.25229315e-11 7.55303486e-01]
x2_opt= [1.00000000e+00 1.00000000e+00 1.21185565e-11 1.00000000e+00
 2.51253897e-11 1.00000000e+00 3.23827442e-11]
x3_opt= [0.46277726 0.353642   0.47196642 0.34704992 0.46908855 0.36512685
 0.46908867]

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -17.5713401261
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.3340198373
            Iterations: 41
            Function evaluations: 82
            Gradient evaluations: 41
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.3523709115
            Iterations: 10
            Function evaluations: 19
            Gradient evaluations: 10
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.2466011146
            Iterations: 16
            Function evaluations: 18
            Gradient evaluations: 16
Optimization Complete
-----------------------------------
z_opt= [0.3385202  0.31586638 0.43013186 0.40091963 0.44948242 0.28447943
 0.66636045]
x1_opt= [0.49914632 0.51219209 0.42397013 0.49640664 1.         0.51357916
 0.46459439]
x2_opt= [0.93309212 0.87109775 0.07542514 0.75197256 0.7947962  0.96550651
 0.07310414]
x3_opt= [0.16991035 0.12273283 0.52641333 0.50616419 0.58234052 0.16718194
 0.26522192]

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -19.9709892626
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -18.7764393391
            Iterations: 29
            Function evaluations: 48
            Gradient evaluations: 29
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -18.7984523514
            Iterations: 13
            Function evaluations: 25
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -18.7760095927
            Iterations: 12
            Function evaluations: 14
            Gradient evaluations: 12
Optimization Complete
-----------------------------------
z_opt= [0.52676523 0.4948946  0.33927549 0.2577856  0.23842374 0.78474961
 0.48018969]
x1_opt= [0.36896262 0.54421918 0.49949421 0.45483266 0.50100481 0.52605053
 0.46873482]
x2_opt= [0.3437708  0.96403831 0.96698873 0.98854472 0.09449379 0.72423685
 0.90587586]
x3_opt= [1.         0.35961938 0.37177107 0.20250902 0.36202729 0.1285527
 0.18895306]

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -22.8080616762
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Iteration limit exceeded    (Exit mode 9)
            Current function value: -18.9683504227
            Iterations: 101
            Function evaluations: 359
            Gradient evaluations: 100
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Iteration limit exceeded    (Exit mode 9)
            Current function value: -18.9683504227
            Iterations: 101
            Function evaluations: 359
            Gradient evaluations: 100
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 13 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations


/home/arnab/Downloads/thesis_project/github_clones/venv/bin/python /home/arnab/Downloads/thesis_project/github_clones/kadmos/examples/scripts/build_database_coupling_density.py
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 24 design variables and 19 response variables (objectives and nonlinear constraints).

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -7.43640664571
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 39 design variables and 34 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.140654629
            Iterations: 24
            Function evaluations: 34
            Gradient evaluations: 24
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.14338494413
            Iterations: 15
            Function evaluations: 22
            Gradient evaluations: 15
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/matplotlib/cbook/deprecation.py:107: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
  warnings.warn(message, mplDeprecation, stacklevel=1)

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 5 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -7.14077343628
            Iterations: 11
            Function evaluations: 12
            Gradient evaluations: 11
Optimization Complete
-----------------------------------
z_opt= [0.69447716 0.40312646 0.20819466 0.43688246 0.80733766 0.61016048]
x1_opt= [0.94300475 0.20827826 0.78450976 1.         0.21230986 0.91183492]
x2_opt= [0.54330825 0.         0.86682154 0.         0.78467001 1.        ]
x3_opt= [0.53312151 0.14755051 0.54692539 0.3334158  0.55654354 0.39705504]

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -10.0565653993
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 44 design variables and 39 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.82651459318
            Iterations: 22
            Function evaluations: 31
            Gradient evaluations: 22
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.82786603873
            Iterations: 14
            Function evaluations: 22
            Gradient evaluations: 14
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 12 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -9.82628764748
            Iterations: 13
            Function evaluations: 14
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
z_opt= [0.60477212 0.31655441 0.42380217 0.71791017 0.49216146 0.37650713]
x1_opt= [0.73460146 0.6951743  0.57477296 0.51499771 0.87640128 0.57477296]
x2_opt= [0.20813294 0.22725341 0.92283615 0.736527   0.77718938 0.88212311]
x3_opt= [0.04766583 0.37881794 0.43291101 0.63459244 0.37806819 0.3297908 ]

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -12.5484001229
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 49 design variables and 44 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -12.1038736563
            Iterations: 19
            Function evaluations: 23
            Gradient evaluations: 19
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -12.1047231021
            Iterations: 13
            Function evaluations: 18
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -12.1036806119
            Iterations: 12
            Function evaluations: 13
            Gradient evaluations: 12
Optimization Complete
-----------------------------------
z_opt= [0.62842075 0.38240258 0.40336902 0.51061006 0.30421728 0.55265527]
x1_opt= [0.48942085 0.26563299 1.         0.265633   1.         0.93369233]
x2_opt= [0.56532517 0.87803999 0.16239513 0.84500464 0.87806723 0.45239652]
x3_opt= [0.18039585 0.36217205 0.62075035 0.46660609 0.11631345 0.36973167]

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -14.9921574755
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 54 design variables and 49 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -14.1956966912
            Iterations: 31
            Function evaluations: 50
            Gradient evaluations: 31
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -14.2112989489
            Iterations: 13
            Function evaluations: 23
            Gradient evaluations: 13
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -14.1953945253
            Iterations: 17
            Function evaluations: 17
            Gradient evaluations: 17
Optimization Complete
-----------------------------------
z_opt= [0.34133585 0.28901611 0.38671474 0.27150467 0.21390018 0.58154591]
x1_opt= [1.       1.       1.       1.       1.       0.906123]
x2_opt= [0.05205264 0.57558731 0.65778826 1.         0.95047885 0.94354205]
x3_opt= [0.44047491 1.         0.22018682 0.28492417 0.50665995 0.13316365]

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -17.7585405027
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 59 design variables and 54 response variables (objectives and nonlinear constraints).
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.2815653099
            Iterations: 38
            Function evaluations: 72
            Gradient evaluations: 38
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.2826398009
            Iterations: 28
            Function evaluations: 59
            Gradient evaluations: 28
Optimization Complete
-----------------------------------
Model viewer data has already has already been recorded for Driver.

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 1 iterations

===
Mda
===
NL: NLBGS Converged in 11 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 10 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 9 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 8 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 7 iterations

===
Mda
===
NL: NLBGS Converged in 6 iterations
Optimization terminated successfully.    (Exit mode 0)
            Current function value: -16.2810682437
            Iterations: 19
            Function evaluations: 34
            Gradient evaluations: 19
Optimization Complete
-----------------------------------
z_opt= [0.8882495  0.28904807 0.30943801 0.47935383 0.74300914 0.32184014]
x1_opt= [0.11356779 0.4150288  1.         0.39809542 1.         0.58638538]
x2_opt= [1.         0.56733176 0.02720432 0.04616868 0.97789054 0.99526247]
x3_opt= [0.18729824 0.02173981 0.11130706 0.32426298 0.33222364 0.02435046]

===
Mda
===
NL: NLBGS Converged in 14 iterations

===
Mda
===
NL: NLBGS Converged in 0 iterations
Iteration limit exceeded    (Exit mode 9)
            Current function value: -20.307935646
            Iterations: 1
            Function evaluations: 1
            Gradient evaluations: 1
Optimization FAILED.
Iteration limit exceeded
-----------------------------------
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/openmdao/core/problem.py:878: RuntimeWarning:Inefficient choice of derivative mode.  You chose 'fwd' for a problem with 64 design variables and 59 response variables (objectives and nonlinear constraints).
/home/arnab/Downloads/thesis_project/github_clones/kadmos/examples/scripts/interpolated_functions.py:752: RuntimeWarning: overflow encountered in double_scalars
  cd = (c_dmin + k * cl ** 2) * fo3
/home/arnab/Downloads/thesis_project/github_clones/kadmos/examples/scripts/interpolated_functions.py:775: RuntimeWarning: overflow encountered in double_scalars
  cd = (c_dmin + k * cl ** 2) * fo3
/home/arnab/Downloads/thesis_project/github_clones/kadmos/examples/scripts/interpolated_functions.py:631: RuntimeWarning: overflow encountered in double_scalars
  cd = (c_dmin + k * cl ** 2) * fo3
/home/arnab/Downloads/thesis_project/github_clones/venv/local/lib/python2.7/site-packages/numpy/core/fromnumeric.py:86: RuntimeWarning: invalid value encountered in reduce
  return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
/home/arnab/Downloads/thesis_project/github_clones/kadmos/examples/scripts/interpolated_functions.py:688: RuntimeWarning: overflow encountered in double_scalars
  cd = (c_dmin + k * cl ** 2) * fo3
/home/arnab/Downloads/thesis_project/github_clones/kadmos/examples/scripts/interpolated_functions.py:1296: RuntimeWarning: invalid value encountered in double_scalars
  range = 661.0 * np.sqrt(theta) * z[2] * fin / sfc * np.log(abs(wt / (wt - wf)))

"""




