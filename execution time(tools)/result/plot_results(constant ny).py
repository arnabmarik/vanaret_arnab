import numpy as np
import pandas as pd
import matplotlib.pylab as plt

counter0 = 1
for ny in [3, 4, 5, 6, 7]:
    fig2 = plt.figure(counter0)
    counter0 += 1
    fig2.suptitle('ny = ' + str(ny), fontsize=8)
    plt.tight_layout()
    result = pd.read_csv("results.csv")
    result = result[result["4.total iterations[MDF]"] < 25]
    print(result)
    counter1 = 0
    for d in [0.3, 0.4, 0.5, 0.6, 0.7]:
        counter1 += 1
        ax = fig2.add_subplot(3, 2, counter1)
        a_mdf = []
        a_idf = []
        for nx in [3, 4, 5, 6, 7]:
            d_idf = result[(result["2.ny"] == ny) & (result["1.nx"] == nx) & (result["3.d"] == d)]["11.total time[IDF_tol]"]
            d_idf = d_idf.to_numpy()
            # remove the outliers
            # d_idf = d_idf[d_idf != max(d_idf)]
            # d_idf = d_idf[d_idf != max(d_idf)]
            a_idf.append(np.median(d_idf))
            ax.scatter((np.ones(np.size(d_idf)) * nx).astype(int), d_idf, marker='o', facecolor='blue',
                       edgecolor='black', linewidths=.2)
            d_mdf = result[(result["2.ny"] == ny) & (result["1.nx"] == nx) & (result["3.d"] == d)]["5.total time[MDF]"]
            d_mdf = d_mdf.to_numpy()
            # remove the outliers
            # d_mdf = d_mdf[d_mdf != max(d_mdf)]
            # d_mdf = d_mdf[d_mdf != max(d_mdf)]
            a_mdf.append(np.median(d_mdf))
            ax.scatter((np.ones(np.size(d_mdf)) * nx).astype(int), d_mdf, marker='*', facecolor='red',
                       edgecolor='black', linewidths=.2)
        plt.xlabel('nx', fontsize=8)
        plt.ylabel('wall time')
        ax.plot(range(3, 8), a_idf)
        ax.plot(range(3, 8), a_mdf)
        if d == 0.3:
            plt.gca().legend(('IDF', 'MDF'))
    plt.tight_layout()
plt.show()