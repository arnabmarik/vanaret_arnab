import pandas as pd
import matplotlib.pylab as plt

fig1 = plt.figure(1)
for i in range(8):
    ax = fig1.add_subplot(4, 2, i+1)
    result = pd.read_csv("result_coupling_density(nx = 7).csv")
    ax.plot(result['d'][i:48:8], result['total time[MDF]'][i:48:8], label='MDF')
    ax.plot(result['d'][i:48:8], result['total time[IDF]'][i:48:8], label='IDF_TOL = 1e-6')
    ax.plot(result['d'][i:48:8], result['total time[IDF_tol]'][i:48:8], label='IDF_TOL = 1e-3')
    if i == 0:
        plt.gca().legend(('MDF', 'IDF_TOL = 1e-6', 'IDF_TOL = 1e-3'), fontsize=5)
    plt.xlabel('d', fontsize=10)
    plt.title('nx = 7, ny = ' + str(i+3), fontsize=10)
    plt.ylabel('wall time')
plt.tight_layout()

fig2 = plt.figure(2)
for i in range(8):
    ax = fig2.add_subplot(4, 2, i+1)
    result = pd.read_csv("result_coupling_density(nx = 6).csv")
    ax.plot(result['d'][i:48:8], result['total time[MDF]'][i:48:8], label='MDF')
    ax.plot(result['d'][i:48:8], result['total time[IDF]'][i:48:8], label='IDF_TOL = 1e-6')
    ax.plot(result['d'][i:48:8], result['total time[IDF_tol]'][i:48:8], label='IDF_TOL = 1e-3')
    if i == 0:
        plt.gca().legend(('MDF', 'IDF_TOL = 1e-6', 'IDF_TOL = 1e-3'), fontsize=5)
    plt.xlabel('d', fontsize=10)
    plt.title('nx = 6, ny = ' + str(i+3), fontsize=10)
    plt.ylabel('wall time')
plt.tight_layout()

fig3 = plt.figure(3)
for i in range(5):
    ax = fig3.add_subplot(3, 2, i+1)
    result = pd.read_csv("result_coupling_density(nx = 5).csv")
    ax.plot(result['d'][i:35:5], result['total time[MDF]'][i:35:5], label='MDF')
    ax.plot(result['d'][i:35:5], result['total time[IDF]'][i:35:5], label='IDF_TOL = 1e-6')
    ax.plot(result['d'][i:35:5], result['total time[IDF_tol]'][i:35:5], label='IDF_TOL = 1e-3')
    if i == 0:
        plt.gca().legend(('MDF', 'IDF_TOL = 1e-6', 'IDF_TOL = 1e-3'), fontsize=5)
    plt.xlabel('d', fontsize=10)
    plt.title('nx = 5, ny = ' + str(i+3), fontsize=10)
    plt.ylabel('wall time')
plt.tight_layout()
plt.show()


