import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# # nx = 3
for d in range(3, 9):
    fig2 = plt.figure(1)
    fig2.suptitle('nx = 3', fontsize=16)
    result = pd.read_csv("result(nx = 3)/result_new_values(." + str(d) + ").csv")
    ax = fig2.add_subplot(3, 2, d - 2)
    a_mdf = []
    a_idf = []
    for ny in range(3, 10):
        d_idf = result[result["ny"] == ny]["total time[IDF_tol]"]
        d_idf = d_idf.to_numpy()
        # remove the outliers
        d_idf = d_idf[d_idf != max(d_idf)]
        d_idf = d_idf[d_idf != max(d_idf)]
        a_idf.append(np.mean(d_idf))
        ax.scatter((np.ones(np.size(d_idf)) * ny).astype(int), d_idf, marker='o', facecolor='blue', edgecolor='black',
                   linewidths=.2)
        d_mdf = result[result["ny"] == ny]["total time[MDF]"]
        d_mdf = d_mdf.to_numpy()
        # remove the outliers
        d_mdf = d_mdf[d_mdf != max(d_mdf)]
        d_mdf = d_mdf[d_mdf != max(d_mdf)]
        a_mdf.append(np.mean(d_mdf))
        ax.scatter((np.ones(np.size(d_mdf)) * ny).astype(int), d_mdf, marker='*', facecolor='red', edgecolor='black',
                   linewidths=.2)
    plt.xlabel('ny', fontsize=8)
    plt.title('d = ' + str(.1 * d))
    plt.ylabel('wall time')
    ax.plot(range(3, len(a_idf) + 3), a_idf)
    ax.plot(range(3, len(a_mdf) + 3), a_mdf)
    if d == 3:
        plt.gca().legend(('IDF', 'MDF'))
plt.tight_layout()

# # nx = 4
for d in range(3, 9):
    fig3 = plt.figure(4)
    fig3.suptitle('nx = 4', fontsize=16)
    result = pd.read_csv("result(nx = 4)/result_new_values(0." + str(d) + ").csv")
    ax = fig3.add_subplot(3, 2, d - 2)
    a_mdf = []
    a_idf = []
    for ny in [3, 4, 6, 7, 8, 9]:
        d_idf = result[result["ny"] == ny]["total time[IDF_tol]"]
        d_idf = d_idf.to_numpy()
        # remove the outliers
        d_idf = d_idf[d_idf != max(d_idf)]
        d_idf = d_idf[d_idf != max(d_idf)]
        a_idf.append(np.mean(d_idf))
        ax.scatter((np.ones(np.size(d_idf)) * ny).astype(int), d_idf, marker='o', facecolor='blue', edgecolor='black',
                   linewidths=.2)
        d_mdf = result[result["ny"] == ny]["total time[MDF]"]
        d_mdf = d_mdf.to_numpy()
        # remove the outliers
        d_mdf = d_mdf[d_mdf != max(d_mdf)]
        d_mdf = d_mdf[d_mdf != max(d_mdf)]
        a_mdf.append(np.mean(d_mdf))
        ax.scatter((np.ones(np.size(d_mdf)) * ny).astype(int), d_mdf, marker='*', facecolor='red', edgecolor='black',
                   linewidths=.2)
    plt.xlabel('ny', fontsize=8)
    plt.title('d = ' + str(.1 * d))
    plt.ylabel('wall time')
    ax.plot([3, 4, 6, 7, 8, 9], a_idf)
    ax.plot([3, 4, 6, 7, 8, 9], a_mdf)
    if d == 3:
        plt.gca().legend(('IDF', 'MDF'))
plt.tight_layout()

# # nx = 5
for d in range(3, 8):
    fig3 = plt.figure(5)
    fig3.suptitle('nx = 5', fontsize=16)
    result = pd.read_csv("result/result_new_values(0." + str(d) + ").csv")
    ax = fig3.add_subplot(3, 2, d - 2)
    a_mdf = []
    a_idf = []
    for ny in range(3, 9):
        d_idf = result[result["ny"] == ny]["total time[IDF_tol]"]
        d_idf = d_idf.to_numpy()
        # remove the outliers
        d_idf = d_idf[d_idf != max(d_idf)]
        d_idf = d_idf[d_idf != max(d_idf)]
        a_idf.append(np.mean(d_idf))
        ax.scatter((np.ones(np.size(d_idf)) * ny).astype(int), d_idf, marker='o', facecolor='blue', edgecolor='black',
                   linewidths=.2)
        d_mdf = result[result["ny"] == ny]["total time[MDF]"]
        d_mdf = d_mdf.to_numpy()
        # remove the outliers
        d_mdf = d_mdf[d_mdf != max(d_mdf)]
        d_mdf = d_mdf[d_mdf != max(d_mdf)]
        a_mdf.append(np.mean(d_mdf))
        ax.scatter((np.ones(np.size(d_mdf)) * ny).astype(int), d_mdf, marker='*', facecolor='red', edgecolor='black',
                   linewidths=.2)
    plt.xlabel('ny', fontsize=8)
    plt.title('d = ' + str(.1 * d))
    plt.ylabel('wall time')
    ax.plot(range(3, len(a_idf) + 3), a_idf)
    ax.plot(range(3, len(a_mdf) + 3), a_mdf)
    if d == 3:
        plt.gca().legend(('IDF', 'MDF'))
plt.tight_layout()

# # nx = 6
for d in range(3, 8):
    fig3 = plt.figure(6)
    fig3.suptitle('nx = 6', fontsize=16)
    result = pd.read_csv("result(nx = 6)/result_new_values(0." + str(d) + ").csv")
    ax = fig3.add_subplot(3, 2, d - 2)
    a_mdf = []
    a_idf = []
    for ny in range(3, 9):
        d_idf = result[result["ny"] == ny]["total time[IDF_tol]"]
        d_idf = d_idf.to_numpy()
        # remove the outliers
        #d_idf = d_idf[d_idf != max(d_idf)]
        #d_idf = d_idf[d_idf != max(d_idf)]
        a_idf.append(np.mean(d_idf))
        ax.scatter((np.ones(np.size(d_idf)) * ny).astype(int), d_idf, marker='o', facecolor='blue', edgecolor='black',
                   linewidths=.2)
        d_mdf = result[result["ny"] == ny]["total time[MDF]"]
        d_mdf = d_mdf.to_numpy()
        # remove the outliers
        #d_mdf = d_mdf[d_mdf != max(d_mdf)]
        #d_mdf = d_mdf[d_mdf != max(d_mdf)]
        a_mdf.append(np.mean(d_mdf))
        ax.scatter((np.ones(np.size(d_mdf)) * ny).astype(int), d_mdf, marker='*', facecolor='red', edgecolor='black',
                   linewidths=.2)
    plt.xlabel('ny', fontsize=8)
    plt.title('d = ' + str(.1 * d))
    plt.ylabel('wall time')
    ax.plot(range(3, len(a_idf) + 3), a_idf)
    ax.plot(range(3, len(a_mdf) + 3), a_mdf)
    if d == 3:
        plt.gca().legend(('IDF', 'MDF'))
plt.tight_layout()

# # nx = 7
for d in range(3, 8):
    fig3 = plt.figure(7)
    fig3.suptitle('nx = 7', fontsize=16)
    result = pd.read_csv("result(nx = 7)/result_new_values(0." + str(d) + ").csv")
    ax = fig3.add_subplot(3, 2, d - 2)
    a_mdf = []
    a_idf = []
    for ny in range(3, 9):
        d_idf = result[result["ny"] == ny]["total time[IDF_tol]"]
        d_idf = d_idf.to_numpy()
        # remove the outliers
        # d_idf = d_idf[d_idf != max(d_idf)]
        # d_idf = d_idf[d_idf != max(d_idf)]
        a_idf.append(np.mean(d_idf))
        ax.scatter((np.ones(np.size(d_idf)) * ny).astype(int), d_idf, marker='o', facecolor='blue', edgecolor='black',
                   linewidths=.2)
        d_mdf = result[result["ny"] == ny]["total time[MDF]"]
        d_mdf = d_mdf.to_numpy()
        # remove the outliers
        # d_mdf = d_mdf[d_mdf != max(d_mdf)]
        # d_mdf = d_mdf[d_mdf != max(d_mdf)]
        a_mdf.append(np.mean(d_mdf))
        ax.scatter((np.ones(np.size(d_mdf)) * ny).astype(int), d_mdf, marker='*', facecolor='red', edgecolor='black',
                   linewidths=.2)
    plt.xlabel('ny', fontsize=8)
    plt.title('d = ' + str(.1 * d))
    plt.ylabel('wall time')
    ax.plot(range(3, len(a_idf) + 3), a_idf)
    ax.plot(range(3, len(a_mdf) + 3), a_mdf)
    if d == 3:
        plt.gca().legend(('IDF', 'MDF'))
plt.tight_layout()

# nx = 10
for d in range(3, 9):
    fig2 = plt.figure(10)
    fig2.suptitle('nx = 10', fontsize=16)
    result = pd.read_csv("result(nx = 10)/result_new_values(." + str(d) + ").csv")
    ax = fig2.add_subplot(3, 2, d - 2)
    a_mdf = []
    a_idf = []
    for ny in range(3, 9):
        d_idf = result[result["ny"] == ny]["total time[IDF_tol]"]
        d_idf = d_idf.to_numpy()
        # remove the outliers
        d_idf = d_idf[d_idf != max(d_idf)]
        d_idf = d_idf[d_idf != max(d_idf)]
        a_idf.append(np.mean(d_idf))
        ax.scatter((np.ones(np.size(d_idf)) * ny).astype(int), d_idf, marker='o', facecolor='blue', edgecolor='black',
                   linewidths=.2)
        d_mdf = result[result["ny"] == ny]["total time[MDF]"]
        d_mdf = d_mdf.to_numpy()
        # remove the outliers
        d_mdf = d_mdf[d_mdf != max(d_mdf)]
        d_mdf = d_mdf[d_mdf != max(d_mdf)]
        a_mdf.append(np.mean(d_mdf))
        ax.scatter((np.ones(np.size(d_mdf)) * ny).astype(int), d_mdf, marker='*', facecolor='red', edgecolor='black',
                   linewidths=.2)
    plt.xlabel('ny', fontsize=8)
    plt.title('d = ' + str(.1 * d))
    plt.ylabel('wall time')
    ax.plot(range(3, len(a_idf) + 3), a_idf)
    ax.plot(range(3, len(a_mdf) + 3), a_mdf)
    if d == 3:
        plt.gca().legend(('IDF', 'MDF'))
plt.tight_layout()

# # nx = 15
for d in range(3, 8):
    fig3 = plt.figure(15)
    fig3.suptitle('nx = 15', fontsize=16)
    result = pd.read_csv("result(nx = 15)/result_new_values(0." + str(d) + ").csv")
    ax = fig3.add_subplot(3, 2, d - 2)
    a_mdf = []
    a_idf = []
    for ny in range(3, 9):
        d_idf = result[result["ny"] == ny]["total time[IDF_tol]"]
        d_idf = d_idf.to_numpy()
        # remove the outliers
        # d_idf = d_idf[d_idf != max(d_idf)]
        # d_idf = d_idf[d_idf != max(d_idf)]
        a_idf.append(np.mean(d_idf))
        ax.scatter((np.ones(np.size(d_idf)) * ny).astype(int), d_idf, marker='o', facecolor='blue', edgecolor='black',
                   linewidths=.2)
        d_mdf = result[result["ny"] == ny]["total time[MDF]"]
        d_mdf = d_mdf.to_numpy()
        # remove the outliers
        # d_mdf = d_mdf[d_mdf != max(d_mdf)]
        # d_mdf = d_mdf[d_mdf != max(d_mdf)]
        a_mdf.append(np.mean(d_mdf))
        ax.scatter((np.ones(np.size(d_mdf)) * ny).astype(int), d_mdf, marker='*', facecolor='red', edgecolor='black',
                   linewidths=.2)
    plt.xlabel('ny', fontsize=8)
    plt.title('d = ' + str(.1 * d))
    plt.ylabel('wall time')
    ax.plot(range(3, len(a_idf) + 3), a_idf)
    ax.plot(range(3, len(a_mdf) + 3), a_mdf)
    if d == 3:
        plt.gca().legend(('IDF', 'MDF'))
plt.tight_layout()
#
# # # nx = 20
for d in [3, 5, 7]:
    fig4 = plt.figure(20)
    fig4.suptitle('nx = 20', fontsize=16)
    result = pd.read_csv("result(nx = 20)/result_new_values(0." + str(d) + ").csv")
    ax = fig4.add_subplot(3, 2, d - 2)
    a_mdf = []
    a_idf = []
    for ny in range(3, 9):
        d_idf = result[result["ny"] == ny]["total time[IDF_tol]"]
        d_idf = d_idf.to_numpy()
        # remove the outliers
        d_idf = d_idf[d_idf != max(d_idf)]
        d_idf = d_idf[d_idf != max(d_idf)]
        a_idf.append(np.mean(d_idf))
        ax.scatter((np.ones(np.size(d_idf)) * ny).astype(int), d_idf, marker='o', facecolor='blue', edgecolor='black',
                   linewidths=.2)
        d_mdf = result[result["ny"] == ny]["total time[MDF]"]
        d_mdf = d_mdf.to_numpy()
        # remove the outliers
        d_mdf = d_mdf[d_mdf != max(d_mdf)]
        d_mdf = d_mdf[d_mdf != max(d_mdf)]
        a_mdf.append(np.mean(d_mdf))
        ax.scatter((np.ones(np.size(d_mdf)) * ny).astype(int), d_mdf, marker='*', facecolor='red', edgecolor='black',
                   linewidths=.2)
    plt.xlabel('ny', fontsize=8)
    plt.title('d = ' + str(.1 * d))
    plt.ylabel('wall time')
    ax.plot(range(3, len(a_idf) + 3), a_idf)
    ax.plot(range(3, len(a_mdf) + 3), a_mdf)
    if d == 3:
        plt.gca().legend(('IDF', 'MDF'))
plt.tight_layout()

plt.show()





