import numpy as np
import pandas as pd
import matplotlib.pylab as plt

counter0 = 1
for ny in [3, 5, 10]:
    fig2 = plt.figure(counter0)
    counter0 += 1
    fig2.suptitle('ny = ' + str(ny), fontsize=8)
    plt.tight_layout()
    result = pd.read_csv("results.csv")
    counter1 = 0
    for d in [0.3, 0.5, 0.8]:
        counter1 += 1
        ax = fig2.add_subplot(3, 1, counter1)
        a_mdf = []
        a_idf = []
        for nx in [4, 10, 15, 20, 25]:
            d_idf = result[(result["ny"] == ny) & (result["nx"] == nx) & (result["d"] == d)]["total time[IDF_tol]"]
            d_idf = d_idf.to_numpy()
            # remove the outliers
            d_idf = d_idf[d_idf != max(d_idf)]
            d_idf = d_idf[d_idf != max(d_idf)]
            a_idf.append(np.median(d_idf))
            ax.scatter((np.ones(np.size(d_idf)) * nx).astype(int), d_idf, marker='o', facecolor='blue',
                       edgecolor='black', linewidths=.2)
            d_mdf = result[(result["ny"] == ny) & (result["nx"] == nx) & (result["d"] == d)]["total time[MDF]"]
            d_mdf = d_mdf.to_numpy()
            # remove the outliers
            d_mdf = d_mdf[d_mdf != max(d_mdf)]
            d_mdf = d_mdf[d_mdf != max(d_mdf)]
            a_mdf.append(np.median(d_mdf))
            ax.scatter((np.ones(np.size(d_mdf)) * nx).astype(int), d_mdf, marker='*', facecolor='red',
                       edgecolor='black', linewidths=.2)
        plt.xlabel('nx', fontsize=8)
        plt.ylabel('wall time')
        ax.plot([4, 10, 15, 20, 25], a_idf)
        ax.plot([4, 10, 15, 20, 25], a_mdf)
        if d == 0.3:
            plt.gca().legend(('IDF', 'MDF'))
    plt.tight_layout()
plt.show()